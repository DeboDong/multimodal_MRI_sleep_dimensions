%% projecting coeff to replication data to evaluate generalizability 
%preparing PCA coeffs of discovery data (step 1-3)

% 1. Load discovery data
load('beh_dis_rep.mat', 'cov_dis')
load('beh_dis_rep.mat', 'sds_dis')
load('beh_dis_rep.mat', 'site_dis')

load('image_data_dis.mat');% image

% Number of principal components to keep will depend on % explained variance
pca_threshold = 50;


% 2. Regress out confounds and obtain regression parameters for the replication dataset

% Regress out confounds from each modality
% Behavior data
clear confounds
confounds = cov_dis;
[sds_dis_reg, coef_mtx_sds, ~, ~] = CBIG_glm_regress_matrix(sds_dis, confounds, 0, []);

% Surface area data
clear confounds
confounds = cov_dis; confounds=[confounds total_surface];
[area_dis_reg, coef_mtx_area, ~, ~] = CBIG_glm_regress_matrix(area_dis, confounds, 0, []);

% Thickness data
clear confounds
confounds = cov_dis; confounds=[confounds total_tiv];
[thickness_dis_reg, coef_mtx_thick, ~, ~] = CBIG_glm_regress_matrix(thickness_dis, confounds, 0, []);

% Volume data
clear confounds
confounds = cov_dis; confounds=[confounds total_tiv];
[volume_dis_reg, coef_mtx_vol, ~, ~] = CBIG_glm_regress_matrix(volume_dis, confounds, 0, []);

% RSFC data
clear confounds
confounds = cov_dis; confounds=[confounds headmotion];
[rest_FC_dis_reg, coef_mtx_fc, ~, ~] = CBIG_glm_regress_matrix(rest_FC_dis, confounds, 0, []);

save ('coef_dis_regression.mat','coef_mtx_sds', 'coef_mtx_area', 'coef_mtx_thick', 'coef_mtx_vol','coef_mtx_fc');

% 3. Compute PCA over imaging data modality and save mean and std for projection
% beh
mu_sds=mean(sds_dis_reg);sigma_sds=std(sds_dis_reg);

% PCA on surface area data
clear pca_scores pca_explained
surf_area_z = zscore(area_dis_reg);mu_surf=mean(area_dis_reg);sigma_surf=std(area_dis_reg);
[surf_area_coeff, pca_scores, ~, ~, pca_explained] = pca(surf_area_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));num_pc_surf = num_pc;
surf_area_pca = pca_scores(:, 1:num_pc);  
mu_surf_pca=mean(surf_area_pca);sigma_surf_pca=std(surf_area_pca);

% PCA on thickness data
clear pca_scores pca_explained
thickness_z = zscore(thickness_dis_reg);mu_thickness=mean(thickness_dis_reg);sigma_thickness=std(thickness_dis_reg);
[thickness_coeff, pca_scores, ~, ~, pca_explained] = pca(thickness_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));num_pc_thickness = num_pc;
thickness_pca = pca_scores(:, 1:num_pc); 
mu_thickness_pca=mean(thickness_pca);sigma_thickness_pca=std(thickness_pca);

% PCA on volume data
clear pca_scores pca_explained
volume_z = zscore(volume_dis_reg);mu_volume=mean(volume_dis_reg);sigma_volume=std(volume_dis_reg);
[volume_coeff, pca_scores, ~, ~, pca_explained] = pca(volume_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1));num_pc_volume = num_pc;
volume_pca = pca_scores(:, 1:num_pc);  
mu_volume_pca=mean(volume_pca);sigma_volume_pca=std(volume_pca);

% PCA on RSFC data 
clear pca_scores pca_explained
rsfc_z = zscore(rest_FC_dis_reg);mu_rsfc=mean(rest_FC_dis_reg);sigma_rsfc=std(rest_FC_dis_reg);
[rsfc_coeff, pca_scores, ~, ~, pca_explained] = pca(rsfc_z, 'Centered', false);
num_pc = (find(cumsum(pca_explained) >= pca_threshold, 1)); num_pc_rsfc=num_pc;
rsfc_pca = pca_scores(:, 1:num_pc);  
mu_rsfc_pca=mean(rsfc_pca);sigma_rsfc_pca=std(rsfc_pca);

% save parameters for projecting
save('num_pca.mat','num_pc_surf', 'num_pc_thickness', 'num_pc_volume', 'num_pc_rsfc')
save('mu_sigma.mat', 'mu_rsfc', 'mu_rsfc_pca', 'mu_sds', 'mu_surf', 'mu_surf_pca', 'mu_thickness', 'mu_thickness_pca', 'mu_volume', 'mu_volume_pca', 'sigma_rsfc', 'sigma_rsfc_pca', 'sigma_sds', 'sigma_surf', 'sigma_surf_pca', 'sigma_thickness', 'sigma_thickness_pca', 'sigma_volume', 'sigma_volume_pca');

% 4. projecting dis into rep
% lold rep
load('image_data_rep.mat');
load('beh_dis_rep.mat', 'sds_rep')
load('beh_dis_rep.mat', 'cov_rep')

load('mu_sigma.mat')
load('num_pca.mat')


% 4.1 Regress out confounds from each modality using regression parameters from the discovery dataset

load('coef_dis_regression.mat');

area_rep =area_dis; thickness_rep=thickness_dis;volume_rep=volume_dis;rsfc_rep=rest_FC_dis;
confounds = cov_rep;
sds_reg_rep = sds_rep - (confounds * coef_mtx_sds(2:end, :)); 

confounds = cov_rep; confounds=[confounds total_surface];
surf_area_reg_rep = area_rep - (confounds* coef_mtx_area(2:end, :)); 

confounds = cov_rep; confounds=[confounds total_tiv];
thickness_reg_rep = thickness_rep - (confounds * coef_mtx_thick(2:end, :)); 

confounds = cov_rep; confounds=[confounds total_tiv];
volume_reg_rep = volume_rep - (confounds * coef_mtx_vol(2:end, :)); 

confounds = cov_rep; confounds=[confounds headmotion];
rsfc_reg_rep = rsfc_rep - (confounds * coef_mtx_fc(2:end, :)); 


% 4.2. Normalize replication dataset using mean and std from the discovery dataset

surf_area_z_rep = (surf_area_reg_rep - mu_surf)./sigma_surf;

thickness_z_rep = (thickness_reg_rep - mu_thickness)./sigma_thickness;

volume_z_rep = (volume_reg_rep - mu_volume)./sigma_volume;

rsfc_z_rep = (rsfc_reg_rep - mu_rsfc)./sigma_rsfc;



% 4.3 Apply PCA coefficients from the discovery dataset to the replication datase
load('discovery_coeff.mat')

surf_area_pca_rep = surf_area_z_rep * surf_area_coeff(:,1:num_pc_surf);

thickness_pca_rep = thickness_z_rep * thickness_coeff(:,1:num_pc_thickness); 

volume_pca_rep = volume_z_rep * volume_coeff(:,1:num_pc_volume);

rsfc_pca_rep = rsfc_z_rep * rsfc_coeff(:,1:num_pc_rsfc);


% 4.4. Normalize projected PCA components using the mean and std of PCA components from the discovery dataset

sigma_X_pcas=[sigma_surf_pca sigma_thickness_pca sigma_volume_pca sigma_rsfc_pca];
mu_X_pcas=[mu_surf_pca mu_thickness_pca mu_volume_pca mu_rsfc_pca];

X_rep_pca = [surf_area_pca_rep thickness_pca_rep volume_pca_rep rsfc_pca_rep];

X_rep_norm = (X_rep_pca - mu_X_pcas)./sigma_X_pcas; 

Y_rep_norm = (sds_reg_rep - mu_sds)./sigma_sds;

% 4.5. generate cross-validated composite scores 

load('pls_analysis.mat', 'U', 'V');

Lx_rep = X_rep_norm * V; 

Ly_rep = Y_rep_norm * U; 

% 4.6 permutation test for LC1/3

% Number of permutations
num_permutations = 10000;
r_permuted = zeros(num_permutations, 1);  

% S1: Compute the observed correlation (actual data)
observed_corr = corr(Lx_rep(:, 1), Ly_rep(:, 1));  % Using the first(1)/third(3) composite score
disp(['Observed correlation: ', num2str(observed_corr)]);

% S2: Perform permutations
for i = 1:num_permutations
    % Permute the behavioral data (Y_rep_norm)
    Y_rep_permuted = Y_rep_norm(randperm(length(Y_rep_norm)), :);  
    
    % Compute the new Ly_rep using the permuted Y_rep data
    Ly_rep_permuted = Y_rep_permuted * U;  
    
    % Compute the correlation for the permuted data
    r_permuted(i) = corr(Lx_rep(:, 1), Ly_rep_permuted(:, 1)); 
end

% S3: Calculate p-value
p_value = sum(abs(r_permuted) >= abs(observed_corr)) / num_permutations;
disp(['Permuted p-value: ', num2str(p_value)]);

