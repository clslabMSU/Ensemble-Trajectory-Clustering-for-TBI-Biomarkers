import numpy as np
from scipy.spatial import distance as distance
from math import pow

class internalIndex:
    def  __init__(self, data, label):
        self.data = np.copy(data)
        self.label = np.copy(label)
        self.data_rows = data.shape[0]
        self.data_cols = data.shape[1]
        self.clusters, self.cluster_count = np.unique(label, return_counts=True)
        self.num_cluster = len(np.unique(label))

# 
    def euclidean_centroid(self, label_num = False):
        if label_num is False:
            return np.mean(self.data, axis=0)
        else:
            cluster_data = self.data[self.label == label_num]
            return np.mean(cluster_data, axis=0)
    def centroid_list(self):
        # Initialize an empty list to store centroids
        c_list = []
        # Iterate over the unique class indices
        for index_i in self.clusters:
            # Find indices where label equals the current class index
            cluster_data = self.data[self.label == index_i]
            # Compute the centroid for the current class
            centroid_i = np.mean(cluster_data, axis=0)
            # Append the centroid to the list
            c_list.append(centroid_i)
        return c_list
    def element_of_clustert (self, label_num):
        return self.data[self.label == label_num]
    def distance_from_cluster(self, cluster_i, centroid_i):
        # Get the elements of the cluster and calculate its centroid
        cluster_elements = self.data[self.label == cluster_i]
        centroid = self.euclidean_centroid(centroid_i)
        # Calculate distances using vectorized operations
        dist = 0
        for row in cluster_elements:
            dist += distance.euclidean(centroid, row)
        return dist
    def distance_from_cluster_sqr(self, cluster_i, centroid_i):
        # Get the elements of the cluster and calculate its centroid
        cluster_elements = self.data[self.label == cluster_i]
        centroid = self.euclidean_centroid(centroid_i)
        # Calculate squared distances using vectorized operations
        dist = 0
        for row in cluster_elements:
            dist += distance.sqeuclidean(centroid, row)
        return dist
    def cluster_stdev(self, i=False):
        if i == 'all':
            result = sum([pow(self.cluster_stdev(c),2) for c in self.clusters])
            return np.sqrt(result / self.num_cluster)
        elif i is not False:
            data_in = self.element_of_clustert(i)
        else:
            data_in = np.copy(self.data)
        var_vec = np.var(data_in, axis=0)
        # return np.sqrt(np.dot(var_vec, var_vec.T))
        return np.sqrt(np.sum(var_vec))
    # def cluster_stdev(self, i=False):
    #     if i is not False:
    #         mx = self.element_of_clustert(i)
    #         centroid = self.euclidean_centroid(i)
    #         #dist = self.distance_from_cluster_sqr(i,i)
    #     else:
    #         mx = np.copy(self.data)
    #         centroid = self.euclidean_centroid(False)
    #     dist = sum([distance.sqeuclidean(centroid, row) for row in mx])
    #     stdev = np.sqrt(dist / (len(mx) ))
    #     return stdev     
    def nn_exclude(self, data_i, label_i, k):
        # search for nearest neighbor from other cluster points
        # Calculate distances between data_i and all other points
        distances = distance.cdist([data_i], self.data)[0]
        # Sort distances and get indices of k nearest neighbors (excluding data_i itself)
        nn_indices = np.argsort(distances)[1:k+1]
        # Count mismatches between label_i and labels of k nearest neighbors
        count = np.sum(self.label[nn_indices] != label_i)
        return count
    def dbi(self):
        centroids = self.centroid_list()
        sum_max_distance = 0
        for i, ii in enumerate(self.clusters):
            intra_distance_i = self.distance_from_cluster(ii, ii) / self.cluster_count[i]
            max_distance_j = 0
            for j, jj in enumerate(self.clusters):
                if i == j:
                    continue
                intra_distance_j = self.distance_from_cluster(jj, jj) / self.cluster_count[j]
                c_distance = (intra_distance_i + intra_distance_j) / distance.euclidean(centroids[i], centroids[j])
                max_distance_j = max(max_distance_j, c_distance)
            sum_max_distance += max_distance_j
        dbi_score = sum_max_distance / self.num_cluster
        return dbi_score
    def xie_benie(self):
        # 
        total_distance = 0
        centroids = self.centroid_list()
        for ii in self.clusters:
            total_distance += self.distance_from_cluster_sqr(ii, ii)
        min_cij = distance.pdist(np.array(centroids), 'sqeuclidean').min()
        xb = total_distance / (self.data_rows * min_cij)
        return xb
    def dunn(self):
        # Dunn's index = min(intra cluster distance) / max(inter cluster distance)
        max_ck = float('-inf')
        for index_k in self.clusters:
            eoc_k = self.element_of_clustert(index_k)
            if len(eoc_k) > 1:
                max_ck = max(max_ck, distance.pdist(eoc_k, 'euclidean').max())
        min_ij_candidate = float('inf')
        for index_i in self.clusters:
            eoc_i = self.element_of_clustert(index_i)
            for index_j in self.clusters:
                if index_i == index_j:
                    continue
                eoc_j = self.element_of_clustert(index_j)
                min_j = float('inf')
                for e_c_i in eoc_i:
                    for e_c_j in eoc_j:
                        min_j = min(min_j, distance.euclidean(e_c_i, e_c_j))
                min_j /= max_ck
                min_ij_candidate = min(min_ij_candidate, min_j)
        return min_ij_candidate
    def CH(self):
        centroids = self.centroid_list()
        data_centroid = self.euclidean_centroid()
        cent_distsqr = 0
        ecent_distsqr = 0
        for i, ii in enumerate(self.clusters):
            n_element_i = self.cluster_count[i]
            sqr_dist = distance.sqeuclidean(centroids[i], data_centroid)
            cent_distsqr += sqr_dist * n_element_i
            ecent_distsqr += self.distance_from_cluster_sqr(ii, ii)
        return (cent_distsqr / (self.num_cluster - 1)) / (ecent_distsqr / (self.data_rows - self.num_cluster))
    def I(self, p = 2):
        """I = (distance_to_center * max_centroid_dist / (number of clusters * compactness)) ** p
            distance_to_center = sum(d(e,c)) where e is row of data, c is data's centroid, d = euclidean distance
            max_centroid_dist = max(d(ci, cj)) where ci, cj is i and j cluster's centroid repectively
            compactness = sum(d(x,ci)) where ci is i cluster'd centroid and x is i cluster row
        Args:
            p (int, optional):  Defaults to 2.

        Returns:
            I: I validation index
        """
        centroids = self.centroid_list()
        data_centroid = self.euclidean_centroid()
        compactness = 0
        for i, ii in enumerate(self.clusters):
            eoc = self.element_of_clustert(ii)
            compactness += np.sum([distance.euclidean(e, centroids[i]) for e in eoc])
        max_centroid_dist = distance.pdist(centroids).max()
        distance_to_center = np.sum([distance.euclidean(e, data_centroid) for e in self.data])
        return pow((max_centroid_dist * distance_to_center) / (self.num_cluster * compactness), p)
    def CVNN(self, k=10):
        COM = 0
        SEP = []
        for i, ii in enumerate(self.clusters):
            eoc = self.element_of_clustert(ii)
            n = self.cluster_count[i]
            if n != 1:
                temp_comp = np.sum(distance.pdist(eoc, metric='euclidean')) * 2 / (n * (n - 1))
                COM += temp_comp
            temp_sep = 0
            for e in eoc:
                q_index_e = self.nn_exclude(e, ii, k)
                # temp_sep += (q_index_e / k) / (n + 1)
                temp_sep += (q_index_e / k) / n
            SEP.append(temp_sep)
        SEP = max(max(SEP), 0.01)
        return COM, SEP
    def CVNN_n(self, COM, SEP,j):
        result = COM[j] / np.max(COM) + SEP[j] / np.max(SEP)
        return result
    
    def Scat(self):
        """Scat = sum(stdev(i)) / stdev(D) / number of clusters
            where   i = i cluster
                    D  = data set
        Returns:
            Scat: Scat
        """
        result = sum([self.cluster_stdev(i) for i in self.clusters])
        return (result / (self.num_cluster  * self.cluster_stdev()))
    def SD_valid(self):
        """calculate Scat and Dis
           Dis = (max_cent_dist / min_cent_dist) * sum(1 / sum(d(ci, cj)))
           max_cent_dist and min_cent_dist is the max and min of clusters centroid distance from each other
           d(ci, cj) = distance btw centroid of cluster i and j

        Returns:
            Scat, Dis and number of clusters 
        """
        scat =self.Scat()
        centroids = self.centroid_list()
        cent_dist = distance.pdist(centroids,'euclidean')
        max_cent_dist =  np.max(cent_dist)
        min_cent_dist = np.min(cent_dist)
        seperation = 0
        for c in centroids:
            seperation = seperation + 1 / sum(distance.cdist([c], centroids)[0])
        dis = seperation * max_cent_dist / min_cent_dist
        
        return scat, dis, self.num_cluster
    
    def SD_valid_n(self, scat, dis, numC, j):
        dis_max = dis[numC.index(max(numC))]
        return dis_max * scat[j] + dis[j]
        #return  dis[-1] * scat[j] + dis[j] 
        
    def f_function(self, cluster_elements, cluster_centroid, stdev):
        f = 0
        for row in cluster_elements:
            if distance.euclidean(row, cluster_centroid) <= stdev:
                f += 1
        return f
    
    def SDbw(self) :
        scat = self.Scat()
        dens_bw = 0
        sij = 0
        centroids = self.centroid_list()
        stdev_all = self.cluster_stdev(i='all')
        for i, ii in enumerate(self.clusters):
            # s_i = 0
            w_eoc_i = 0
            eoc_i = self.element_of_clustert(ii)
            stdev_i = self.cluster_stdev(ii)
            w_eoc_i = self.f_function(cluster_elements=eoc_i, cluster_centroid=centroids[i], stdev=stdev_all)
            for j, jj in enumerate(self.clusters):
                if i == j:
                    continue
                else:
                    s_j = 0
                    w_eoc_j = 0
                    eoc_j = self.element_of_clustert(jj)
                    stdev_j = self.cluster_stdev(jj)
                    w_eoc_j = self.f_function(cluster_elements=eoc_j, cluster_centroid=centroids[j], stdev=stdev_all)
                    weight = max(w_eoc_i, w_eoc_j)
                    eoc_ij = np.concatenate((eoc_i, eoc_j), axis=0)
                    # centroid_u_ij = np.mean(eoc_ij, axis=0)
                    centroid_u_ij = (centroids[i] + centroids[j]) / 2
                    # ij_var = np.var(a=eoc_ij, axis=0)
                    # stdev_ij = np.sqrt(np.sum(ij_var))
                    dist_ij = sum([distance.sqeuclidean(centroid_u_ij, row) for row in eoc_ij])
                    stdev_ij = np.sqrt(dist_ij / (len(eoc_ij)))
                    weight_ij = 0
                    weight_ij = self.f_function(cluster_elements=eoc_ij, cluster_centroid=centroid_u_ij, stdev=stdev_all)
                    s_j += (weight_ij / weight)
            sij += s_j
        dens_bw = sij / (self.num_cluster * (self.num_cluster - 1))
        return (scat + dens_bw)    
    def mia(self):
        total_distance = 0
        for ii in self.clusters:
            total_distance += self.distance_from_cluster_sqr(ii, ii)
        return np.sqrt(total_distance / self.num_cluster)
    def smi(self):
        centroids = self.centroid_list()
        smi_score = float('-Inf')
        for i, ii in enumerate(self.clusters):
            # for j, jj in enumerate(self.clusters):
            for j in range(i):
                # if i <= j:
                #     continue
                ln_dist = np.log(distance.euclidean(centroids[i], centroids[j]))
                smi_score = max(1 / (1 - (1 / ln_dist)), smi_score)
        return smi_score
                
    def new_dbi(self):
        centroids = self.centroid_list()
        sum_distance = 0
        for i, ii in enumerate(self.clusters):
            intra_distance_i = self.distance_from_cluster(ii, ii) / self.cluster_count[i]
            sum_distance_j = 0
            for j, jj in enumerate(self.clusters):
                if i >= j:
                    continue
                intra_distance_j = self.distance_from_cluster(jj, jj) / self.cluster_count[j]
                c_distance = (intra_distance_i + intra_distance_j) / distance.euclidean(centroids[i], centroids[j])
                sum_distance_j +=  c_distance
            sum_distance += sum_distance_j
        new_dbi_score = sum_distance / (self.num_cluster * (self.num_cluster - 1))
        return new_dbi_score
        
        


        