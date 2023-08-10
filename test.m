
%% Test MIRT basic recon functions

%% Setup MIRT

clear all
% ir_mex_build  % Only for windows, use once
% cd 'D:\MATLAB\MIRT_fessler\mirt\'
% setup

%% Read example image

fov = 250;  % FOV in mm
image0 = imread('mirt\data\downloads\mribrain.jpg', 'jpg');
I0 = image0(1:4:end, 1:4:end);

%% Generate kspace traj and Gmri object

% Uniform FFT
traj_type = 'half+8';
N = size(I0);
mask = true(N);
[kspace, omega, wi_traj] = mri_trajectory(traj_type, {}, ...
    N, fov, {'voronoi'});
Am = Gmri(kspace, mask, 'fov', fov);

%
J = [6 6];
nufft_args = {N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb'};
% Am = Gmri(kspace, mask, 'fov', fov, 'nufft', nufft_args);

%% Simulated kspace data

K = fftshift(ifft2(fftshift(I0)));
switch traj_type
    case 'cartesian'
        kspace = K(:);
    case 'half+8'
        kspace = K(:,1:N(1)/2+1+8);
        kspace = kspace(:);
end

%% Recon

wi_basis = wi_traj ./ Am.arg.basis.transform;

printm 'conj. phase reconstruction'
xcp = Am' * (wi_basis .* kspace);
xcp = embed(xcp, mask);
subplot(1,2,1)
imshow(abs(xcp),[0,max(max(abs(xcp)))]);
max(max(abs(xcp)))

printm 'PCG with quadratic regularizer'
beta = 2^-7 * size(omega,1);
R = Reg1(mask, 'beta', beta);
C = R.C;
niter = 10;
xpcg = qpwls_pcg(0*xcp(:), Am, 1, kspace(:), 0, C, 1, niter);
xpcg = embed(xpcg(:,end), mask);
subplot(1,2,2)
imshow(abs(xpcg),[0,max(max(abs(xpcg)))]);
max(max(abs(xpcg)))
