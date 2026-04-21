%% Multimodal PLS 
% adapted from https://github.com/valkebets/multimodal_psychopathology_components/blob/main/analyses/multimodal_PLS.m
% 
% 1. Load data
% 2. Regress out confounds
% 3. Reduce imaging data dimensionality with PCA
% 4. PLS analysis 

% download subfunctions from https://github.com/ThomasYeoLab/CBIG/tree/master/external_packages/matlab/non_default_packages/PLS_MIPlab
addpath('..../PLS_MIPlab/')
% 1. Load data
load('beh_dis_rep.mat', 'cov_dis')
load('beh_dis_rep.mat', 'sds_dis')
load('beh_dis_rep.mat', 'site_dis')
% Load imaging data
load('image_data_dis.mat');

% Number of principal components to keep will depend on % explained variance
pca_threshold = 50;


% 2. Regress out confounds 

% Regress out confounds from each modality

% Behavior data
clear confounds
confounds = cov_dis;
[sds_dis_reg, ~, ~, ~] = CBIG_glm_regress_matrix(sds_dis, confounds, 0, []);

% Surface area data
clear confounds
confounds = cov_dis; confounds=[confounds total_surface];
[area_dis_reg, ~, ~, ~] = CBIG_glm_regress_matrix(area_dis, confounds, 0, []);

% Thickness data
clear confounds
confounds = cov_dis; confounds=[confounds total_tiv];
[thickness_dis_reg, ~, ~, ~] = CBIG_glm_regress_matrix(thickness_dis, confounds, 0, []);

% Volume data
clear confounds
confounds = cov_dis; confounds=[confounds total_tiv];
[volume_dis_reg, ~, ~, ~] = CBIG_glm_regress_matrix(volume_dis, confounds, 0, []);

% RSFC data
clear confounds
confounds = cov_dis; confounds=[confounds headmotion];
[rest_FC_dis_reg, ~, ~, ~] = CBIG_glm_regress_matrix(rest_FC_dis, confounds, 0, []);


% 3. Compute PCA over imaging data modality

% PCA on surface area data
clear pca_scores pca_explained
surf_area_z = zscore(area_dis_reg);
[~, pca_scores, ~, ~, pca_explained] = pca(surf_area_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));
surf_area_pca = pca_scores(:, 1:num_pc);  

% PCA on thickness data
clear pca_scores pca_explained
thickness_z = zscore(thickness_dis_reg);
[~, pca_scores, ~, ~, pca_explained] = pca(thickness_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));
thickness_pca = pca_scores(:, 1:num_pc);  

% PCA on volume data
clear pca_scores pca_explained
volume_z = zscore(volume_dis_reg);%
[~, pca_scores, ~, ~, pca_explained] = pca(volume_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));
volume_pca = pca_scores(:, 1:num_pc);  

% PCA on RSFC data 
clear pca_scores pca_explained
rsfc_z = zscore(rest_FC_dis_reg);
[~, pca_scores, ~, ~, pca_explained] = pca(rsfc_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));
rsfc_pca = pca_scores(:, 1:num_pc);  
% 4. PLS analysis
    
Y = sds_dis_reg;
X = [surf_area_pca thickness_pca volume_pca rsfc_pca];

[U, S, V, Lx, Ly, explCovLC, behav_loadings, pca_img_loadings] = ...
    myPLS_analysis(X, Y, 1, 1);
save('pls_analysis.mat','U', 'S', 'V', 'Lx', 'Ly', 'explCovLC', 'behav_loadings', 'pca_img_loadings');

% Re-compute loadings in original space
surf_area_loadings = corr(Lx, area_dis_reg);
thickness_loadings = corr(Lx, thickness_dis_reg);
volume_loadings = corr(Lx, volume_dis_reg);
RSFC_loadings = corr(Lx, rest_FC_dis_reg);
% save original loading
save('original_loadings.mat','surf_area_loadings','thickness_loadings','volume_loadings','RSFC_loadings');

% Permutation testing (while accounting for site)

load('beh_dis_rep.mat', 'site_dis');
site_unique=unique(site_dis);
for i=1:length(site_unique)
    site_dis(find(site_dis==site_unique(i,1)),1)=i;
end
pvals_LC = myPLS_permut(X, Y, U, S, 10000, site_dis, 1, 1, 1)
pvaluesBH = mafdr(pvals_LC(1:5,1), 'BHFDR', true);
