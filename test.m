
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

% Visualize kspace traj (with same direction as imshow/matrix display)
figure
plot(omega(:,2), omega(:,1))
axis([-pi pi -pi pi])
set(gca,'Ydir','reverse')
xlabel('Y/Phase Encoding')
ylabel('X/Frequency Encoding')

%% Simulated kspace data

K0 = fftshift(fft2(fftshift(I0)));
K = K0 + 0.01 * complex(randn(size(K0)), randn(size(K0)));
switch traj_type
    case 'cartesian'
        K = K(:);
    case 'cart:y/2'
        K = K(:,1:2:end);
        K = K(:);
    case 'half+8'
        K = K(:,1:N(1)/2+1+8);
        K = K(:);
end

%% Recon

wi_basis = wi_traj ./ Am.arg.basis.transform;

printm 'conj. phase reconstruction'
xcp = Am' * (wi_basis .* K);
xcp = embed(xcp, mask);
subplot(1,2,1)
imshow(abs(xcp),[0,max(max(abs(xcp)))]);
title('conj. phase reconstruction');

printm 'PCG with quadratic regularizer'
beta = 2^-7 * size(omega,1);
R = Reg1(mask, 'beta', beta);
C = R.C;
niter = 10;
xpcg = qpwls_pcg(0*xcp(:), Am, 1, K(:), 0, C, 1, niter);
xpcg = embed(xpcg(:,end), mask);
subplot(1,2,2)
imshow(abs(xpcg),[0,max(max(abs(xpcg)))]);
title('PCG with quadratic regularizer');
