%% Movie Feature Analysis
%
% Date Created: 21 June 2023
% Last Modified: 20 February 2025
% Created By: Morgan Fogarty (m.fogarty@wustl.edu)
%
% This script uses movie viewing data from the Gates system paper to create
% regressor maps. This is meant as a tutorial script for movie regressor
% analysis and focuses on single subject analysis. This can be expanded to
% work for group average maps. 
%
% The steps of this script include:
%   1. Load and prepare data
%   2. Load and prepare regressors (convolve with hrf and filtering)
%   3. Correlation analysis approach (simple method)
%   4. GLM analysis approach (recommended method)
%
% This script serves as an example for how to generate feature regressor
% maps using a univariate (correlation) appraoch and a multivariate
% (general linear model) approach.

%% Settings:
addpath(genpath('D:\NeuroDOT')) %NeuroDOT path
 addpath(genpath('D:\movieAnalysisTutorial\FeatureRegressors'));
% addpath(genpath('/data/culver/data1/matlab_codes/NeuroDOT_Internal_Additional_Files_and_Functions')) %NeuroDOT support files path
% addpath('/data/culver/data1/Morgan/Scripts/movieAnalysisTutorial'); %Path with tutorial script
% addpath(genpath('/data/culver/data1/dot_gates/regressors/regressor_files/Stim_comp_regressors/')); % This is where the regressors are stored 
% addpath('/data/culver/data1/dot_gates/scripts/'); %path to Gates scripts

% Choose parameters for 
GVTDth = 5.0e-4; % set the GVTD threshold, for more information on GVTD see: doi/10.1002/hbm.25111
HbO_chromophoreID = 1; % Choose between oxyhemoglobin (1) or deoxyhemoglobin (2) 

% Support files
load('hrf_DOT3.mat'); % hemodynamic response function from NeuroDOT
[~,infoMNI]=LoadVolumetricData('mni152nl_T1_on_333',[],'4dfp'); %MNI info file, from NeuroDOT
load('MNI164k_big.mat'); % MNI surface files, from NeuroDOT

%% Loading and Preparing Data:
% The movie viewing data for adults from the Gates paper can be found here: 
load('MovieParticipants_Adults.mat')

dataDirectory= 'D:\HBM_45_e26684_TripathyFogarty\Data\'; %This is where the data is stored

% Select a subject from movieRunPairs
selectedMovieRun = 1; 
subject = char(MovieParticipants{selectedMovieRun,1});
sess = num2str(MovieParticipants{selectedMovieRun,2});
run = char(MovieParticipants{selectedMovieRun,3});
clipname = char(MovieParticipants{selectedMovieRun,5});

% Load the info file:
m1= load([dataDirectory,'\subj-',subject,'\sess-',sess,'\subj-',subject,'-sess-',sess,'-run-',run,'.mat']);

% Load Data: 
% m1 = LoadVolumetricData(['subj-',subject,'-sess-',sess,'-run-',run,'_HbO'],[dataDirectory,'\subj-',subject,'\sess-',sess,'\'],'nii.gz');

% Correct synch points based on audio
% [m1.info, movieStartTime] = movieSynchRealign(m1.info,clipname);

% Crop the timetraces to the start and stop of the movie
mov001 = m1(:,m1.info.paradigm.synchpts(2):m1.info.paradigm.synchpts(3)-1); 

% Lowpass filter the data
mov001 = lowpass(mov001,0.1,1); 

% Reshape into a volume and transform to MNI space
mov001_reshape = Good_Vox2vol(mov001,m1.info.tissue.dim);
mov001_reshape_MNI = affine3d_img(mov001_reshape,m1.info.tissue.dim,infoMNI,affine);

% GVTD Filtering: make a vector of which timepoints to censor 
keep=m1.info.GVTD_filt_rs(m1.info.paradigm.synchpts(2):m1.info.paradigm.synchpts(3)-1)<GVTDth;

%% Load and Prepare Feature Regressors:
% All regressors are stored here: '/data/culver/data1/dot_gates/regressors/regressor_files/Stim_comp_regressors/'

% Load Regressor Files: 
load([clipname '_d_lum.mat'])
load([clipname '_env.mat'])
load([clipname '_lum.mat'])
load([clipname '_audio_coding_consensus.mat'])
load([clipname '_face_coding_consensus.mat'])

duration = size(mov001,2); % Store movie clip duration for cropping regressors

% Convole regressors with hemodynamic response function, z-score, and crop duration
% to the length of the movie viewing data if necessary:
env_hdr = conv(env,hrf);
env_hdr = zscore(env_hdr(1:duration));
env_hdr=env_hdr';

lum_hdr = conv(lum_rs,hrf);
lum_hdr = zscore(lum_hdr(1:duration));

d_lum_hdr = conv(d_lum_rs,hrf);
d_lum_hdr = zscore(d_lum_hdr(1:duration));

speech_hdr = conv(language_consensus,hrf);
speech_hdr = zscore(speech_hdr(1:duration));

face_hdr = conv(faces_consensus,hrf);
face_hdr = zscore(face_hdr(1:duration));
face_hdr=face_hdr';

d_lum_square_hdr = conv(d_lum_square_rs,hrf);
d_lum_square_hdr = zscore(d_lum_square_hdr(1:duration));

% Group regressors
regressors(1,:) = env_hdr;
regressors(2,:) = speech_hdr;
regressors(3,:) = lum_hdr;
regressors(4,:) = d_lum_hdr;
regressors(5,:) = face_hdr;
regressors(6,:) = d_lum_square_hdr;

regressorNames = {'env','speech','lum','dlum','face','dlumSquare'};

max_num_regressors = size(regressors,1);

% match the highpass and lowpass filtering of the data:
regressors = highpass(regressors,0.02,1);
regressors = lowpass(regressors,0.1,1); 

%% Regressor Analysis: Correlation Method (simple)
% The correlation analysis involves simply taking the correlation of each
% voxel and our regressor. This is done using 'corr'. 

for regressor_no=1:max_num_regressors
    % Compute the correlation
    regressorMap = corr(mov001(:,keep)',regressors(regressor_no,keep)');
    regressorMap_reshape = Good_Vox2vol(regressorMap,m1.info.tissue.dim);
    regressorMap_MNI = affine3d_img(regressorMap_reshape,m1.info.tissue.dim,infoMNI,affine); 
    
    % Plot the regressor activation maps
    params.Scale = 0.5;
    params.Th.P = 0; params.Th.N = 0;
    params.ctx='std';
    PlotInterpSurfMesh(regressorMap_MNI,MNIl,MNIr,infoMNI,params);
    title([regressorNames{regressor_no}],'Color','w');
end

%% Regressor Analysis: GLM Method (recommended) 
% The GLM approach helps to separate the regressors by plotting beta value
% maps as opposed to simple correlation maps. This approach was successful
% for parsing apart the face and speech regressor responses. For this
% approach, we need to include the regressors in our design matrix which is
% a time x events matrix. This matrix will also include GVTD filtering. 

% Set GLM parameters:
GLMparams.Nuisance_Reg = 1;
GLMparams.Nuisance_Regs = regressors'; %setting the regressors as the nuisance regressors
GLMparams.GVTD_censor=1;
GLMparams.GVTD_th = GVTDth;

% adjusting the paradigm info so it just includes the movie viewing
% timepoints: 
GLMinfo = m1.info;
GLMinfo.paradigm.synchtype = m1.info.paradigm.synchtype(2:3);
GLMinfo.paradigm.synchpts = m1.info.paradigm.synchpts(2:3);
GLMinfo.paradigm.synchpts(2) = GLMinfo.paradigm.synchpts(2) - GLMinfo.paradigm.synchpts(1);
GLMinfo.paradigm.synchpts(1) = 1;
GLMinfo.paradigm.Pulse_1 = 2;
GLMinfo.paradigm.Pulse_2 = 1;
GLMinfo.GVTD_filt_rs = m1.info.GVTD_filt_rs((round(movieStartTime)):end-1);

% GLM regressor analysis:
[b1,e1,DM1,EDM1]=GLM_181206(mov001,hrf,GLMinfo,GLMparams); 

for regressor_no = 1:max_num_regressors
    bThisRegressor = squeeze(b1(:,(3+regressor_no)))-squeeze(b1(:,2)); %get the beta values for our regressor of interest
    bThisRegressorVol=Good_Vox2vol(bThisRegressor,m1.info.tissue.dim);
    bThisRegressorVol_MNI=affine3d_img(bThisRegressorVol,m1.info.tissue.dim,infoMNI,affine);
    
    % Plot the regressor activation maps
    clear params
    params.Th.P = 0; params.Th.N = 0;
    params.ctx='std';
    PlotInterpSurfMesh(bThisRegressorVol_MNI,MNIl,MNIr,infoMNI,params);
    title([regressorNames{regressor_no}],'Color','w');
end

