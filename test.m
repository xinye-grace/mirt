
%% Test MIRT basic recon functions

%% Setup MIRT

clear all
% ir_mex_build  % Only for windows, use once
% cd 'D:\MATLAB\MIRT_fessler\mirt\'
setup

%% Read example image

fov = 250;  % FOV in mm
image0 = imread('mirt\data\downloads\mribrain.jpg', 'jpg');
image_down2 = image0(1:2:end, 1:2:end);
image_down4 = image0(1:4:end, 1:4:end);

%% Simulated kspace data (Cartesian)

I0 = image_down4;
K = fftshift(ifft2(fftshift(I0)));
kspace_cartesian = K(:);
clear tmp

%% Generate kspace traj and Gmri object

traj_type = 'cartesian';
J = [6 6];

N = size(I0);
nufft_args = {N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb'};
mask = true(N);

[kspace, omega, wi_traj] = mri_trajectory(traj_type, {}, ...
    N, fov, {'voronoi'});

Am = Gmri(kspace, mask, 'fov', fov);
% Am = Gmri(kspace, mask, 'fov', fov, 'nufft', nufft_args);

%% Recon

wi_basis = wi_traj ./ Am.arg.basis.transform;

printm 'conj. phase reconstruction'
xcp = Am' * (wi_basis .* kspace_cartesian);
xcp = embed(xcp, mask);
subplot(1,2,1)
imshow(abs(xcp),[0,max(max(abs(xcp)))]);
max(max(abs(xcp)))

printm 'PCG with quadratic regularizer'
beta = 2^-7 * size(omega,1);
R = Reg1(mask, 'beta', beta);
C = R.C;
niter = 10;
xpcg = qpwls_pcg(0*xcp(:), Am, 1, kspace_cartesian(:), 0, C, 1, niter);
xpcg = embed(xpcg(:,end), mask);
subplot(1,2,2)
imshow(abs(xpcg),[0,max(max(abs(xpcg)))]);
max(max(abs(xpcg)))
