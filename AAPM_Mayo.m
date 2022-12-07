%%��Ҫ�ο����£���An Anisotropic Fourth-Order Diffusion Filter for Image Noise Removal 2011 Mohammad��
%%���£���An adaptive diffusion coefficient selection for image denoising 2017 Hossein��
%%���£���Artifact Suppressed Nonlinear Diffusion Filtering for Low-Dose CT Image Processing 2019 YI LIU��
%%�㷨ԭ�����û�ȡ�Ĳв�ͼ����òв�ֲ�����(�в�ֲ�����)�����ͼ���ݶ�Ĥ�����¶�����ɢ����C(s),����������αӰ������£����õı���ͼ��ϸ�ںͱ�Ե;
%%����Ľ׸�������ƫ΢�ַ��̣��ڱ�Ե������������ɢ����ƽ��������ǿ��ɢ�����ƣ���һ����������αӰ�ͱ���ͼ���Ե��
%% �˴�����Ҫ����AAPM-Mayo-CT-Challengeģ��ͼ���CTͼ��
function main
close all
clear all
clc

addpath(genpath(pwd))

%L014 ��ǻͼ��
% fileName_Nonoise = '.\pic\Abdomen\L014_Full_38.png';
% fileName_Noise = '.\pic\Abdomen\L014_Quarter_38.png';

%L006 ��ǻͼ��
% fileName_Nonoise = '.\pic\Chest\L019_Full_8.png';
% fileName_Noise = '.\pic\Chest\L019_Quarter_8.png';

%L006 ��ǻͼ��
fileName_Nonoise = '.\pic\Pelvic\L006_Full_66.png';
fileName_Noise = '.\pic\Pelvic\L006_Quarter_66.png';

g= double(rgb2gray(imread(fileName_Nonoise)));                        % ������ͼ�� AAPM-Mayo-CT-Challenge�ͼ���CTͼ��
I_noi = double(rgb2gray(imread(fileName_Noise)));                     % ����ͼ��

figure,imshow(I_noi,[],'border','tight');title('�ͼ���CTͼ��');
imcontrast;
figure,imshow(g,[],'border','tight');title('��׼����CTͼ��');
imcontrast;

% ��������ָ��
MSSIM_a = mssim(g, I_noi);
fprintf('�ṹ���ƶ�Ϊ��%.4f\n',MSSIM_a);
Psnr_a = Cal_Psnr(g, I_noi);
fprintf('��ֵ�����Ϊ��%.4f\n',Psnr_a);
[FSIMa, FSIMca ] = FeatureSIM (g, I_noi);
fprintf('�������ƶ�Ϊ��%.4f\n',FSIMa);
gmsda = GMSD(g, I_noi)   ;
fprintf('����������ƫ��Ϊ��%.4f\n',gmsda);                  %GMSD �ݶȷ���������ƫ�� ��ֵԽСԽ�ã�

I_Fide = I_noi;

%��ɢ��ֵ�ļ���
[Gx,Gy]=gradient(I_noi);
gradientimage = sqrt(Gx .* Gx + Gy .* Gy);
gradientimage1=gradientimage(:);
ind= find(gradientimage1>0);
kk=gradientimage1(ind);
kk=kk(:);
len=size(kk,1);
kc=sum(kk)/len*0.9;

% ��������
niter = 20000;   %�Ľ�ƫ΢�ַ��̵ĵ�������������ssim����������ٴε���
del_t = 0.03;    %ʱ�䲽��
beta = 1.8;      % ��������ϵ��

% ��ǻͼ��
% Alpha = 2.0;     % �в�����Ȩ��ϵ��
% Lambada = 1;  % ������Ȩ��ϵ��  0.6

% ��ǻͼ��
% Alpha = 4.6;     % �в�����Ȩ��ϵ��
% Lambada = 0.6;  % ������Ȩ��ϵ��

% ��ǻͼ��
Alpha = 2.2;     % �в�����Ȩ��ϵ��
Lambada = 0.5;  % ������Ȩ��ϵ��


%%��ͼ��Ĳв�
% sub = I_noi - g;
% sigma = var(sub(:));      %��ԭ���ĵ���˼Ϊ��˹��������������׼��ͼ���CTͼ��Ϊ����αӰ�Ͱߵ�����
% sigma_sq = sigma^2;

I_noi = gpuArray(I_noi);  %GPU����
g = gpuArray(g);  %GPU����

tic
% �в�ͼ��Ͳв�ֲ�������ȡ
[PR_ENER] = Hoss_PR(I_noi,kc,beta);
%%���������Ľ�ƫ΢��ȥ��
I_Out = Ada_ForthOrder_PR(g, I_noi, I_Fide, PR_ENER, del_t, niter, kc, Alpha, Lambada);
toc

% �����ݴ�GPU�ŵ�CPU
g=gather(g);
I_out = gather(I_Out);
I_noi = gather(I_noi);

% ��������ָ��
MSSIM_a = mssim(g, I_out);
fprintf('�ṹ���ƶ�Ϊ��%.4f\n',MSSIM_a);
Psnr_a = Cal_Psnr(g, I_out);
fprintf('��ֵ�����Ϊ��%.4f\n',Psnr_a);
[FSIMa, FSIMca ] = FeatureSIM (g, I_out);
fprintf('�������ƶ�Ϊ��%.4f\n',FSIMa);
gmsda = GMSD(g, I_out)   ;
fprintf('����������ƫ��Ϊ��%.4f\n',gmsda);                  %GMSD �ݶȷ���������ƫ�� ��ֵԽСԽ�ã�

figure,imshow(I_out,[],'border','tight');
imcontrast;

end

%����ͼ��в���Ϣ
function [PR_ENER] = Hoss_PR(I_noi,Kc,beta)
%%���룺I_oriΪ������ͼ��I_noiΪ��������ͼ��KcΪ��ɢ��������ɢ�theta_sqΪͼ���������betaΪ���������Ʋв��е���Ϣ����
%%�����PR_ImΪ����ͼ������������Ľ���ɢ��Ĳ�ֵͼ��PR_ENERΪ�в�ֲ�������theta_sq_norΪ�в�ֲ�����
Gauss_He_n=fspecial('gaussian',5,5);
Assump =imfilter(I_noi, Gauss_He_n);
sub_Ass = I_noi-Assump;
theta_sq = var(sub_Ass(:));
thread_var = beta * theta_sq;
Var_Ir = 0; Time_Forth = 0;
ds = 1; % ͼ�����չ��Ե
kc = Kc; % ���������Ľ���ɢ����ֵ
del_t = 0.03; %�Ľ׸������Ե�������
I_Noi_re = I_noi;
[M,N] = size(I_noi);
%%��һ�����Ƚ����Ľ׸���������ɢ����ȡͼ��Ĳ�
while(Var_Ir < thread_var && Time_Forth<20000)
    
    % ��1��������ͼ���ڷ��ߣ���ֱ��Ե�������ߣ����ű�Ե���򣩵Ķ���ƫ����
    diff_1 =padarray(I_noi,[ds,ds],'symmetric','both');  %ͼ���Ե����
    % North, South, East and West pixel
    deltaN = diff_1(1:M,  2:N+1);
    deltaS = diff_1(3:M+2,2:N+1);
    deltaE = diff_1(2:M+1,3:N+2);
    deltaW = diff_1(2:M+1,  1:N);
    
    deltaNW = diff_1(1:M,   1:N);
    deltaSE = diff_1(3:M+2, 3:N+2);
    deltaNE = diff_1(1:M,   3:N+2);
    deltaSW = diff_1(3:M+2, 1:N);
    
    Ux = (deltaN - deltaS)./2;
    Uy = (deltaW - deltaE)./2;
    Uxx = deltaN + deltaS - 2.*I_noi;
    Uyy = deltaW + deltaE - 2.*I_noi;
    Uxy= (deltaNW + deltaSE - deltaNE -deltaSW)./4;   %% ����һ�׶���ƫ΢������
    
    Ux_2 = Ux.^2;
    Uy_2 = Uy.^2;
    Sum_xy2 = Ux_2 + Uy_2;
    Tidu_Mo = Sum_xy2.^0.5;
    Dnn = (Uxx.*Ux_2 + 2.*Ux.*Uy.*Uxy + Uyy.*Uy_2)./(Sum_xy2 + eps);   %% 4�׸���������ɢ���¹�ʽ14
    Dtt = (Uxx.*Uy_2 - 2.*Ux.*Uy.*Uxy + Uyy.*Ux_2)./(Sum_xy2 + eps);   %%
    
    % ��2�������� c(|deltaU|) ��ɢ������ʽ5
    Cs = kc^2./(kc^2 + (Tidu_Mo).^2);   %% ��ʽ��5���Ľ��
    
    %��3�������� Dxx_yy��������˹����
    P_Dxx_yy = Cs.^2.*Dnn + Cs.*Dtt;              %%% �Դ���������˹����
    diff_2 = padarray(P_Dxx_yy,[ds,ds],'symmetric','both');  %%�����Ե
    
    %  North, South, East and West pixel
    g_deltaN = diff_2(1:M,  2:N+1);
    g_deltaS = diff_2(3:M+2,2:N+1);
    g_deltaE = diff_2(2:M+1,3:N+2);
    g_deltaW = diff_2(2:M+1,  1:N);
    L_Dxx_yy = (g_deltaN + g_deltaS + g_deltaE + g_deltaW) - 4 * P_Dxx_yy;
    
    % �����㴦����
    I_noi = I_noi -del_t*L_Dxx_yy;
    
    PR_Im = I_Noi_re - I_noi;
    mean_Ir=mean2(PR_Im);   % ��ֵ
    
    R=(PR_Im-mean_Ir).^2;
    Var_Ir=sum(sum(R))/(M*N);
    
    Time_Forth = Time_Forth +1;
    %     sprintf('����Ϊ��%f',Var_Ir)
    %
end
fprintf('��в�ͼ���������Ϊ��%d\n',Time_Forth);
%figure;imshow(PR_Im,[]); title('�Ľ׸���������ɢ�в�ͼ��');

%%
% �ڶ���������в�ֲ�����
filter_width=9;
PR_Im =padarray(PR_Im,[filter_width,filter_width],'symmetric','both');  %ͼ���Ե����

PR_Im =medfilt2(PR_Im,[3,3]);
mean_he=fspecial('average',[filter_width,filter_width]);%��ֵ�˲�
IR_mean=imfilter(PR_Im,mean_he); %�Բв�ͼ����оֲ���ֵ
P=(PR_Im-IR_mean).^2;

gaus_he=fspecial('gaussian',filter_width,5);
LocalPower =imfilter(P, gaus_he);
PR_ENER=LocalPower((1+filter_width):(M+filter_width),(1+filter_width):(N+filter_width));

figure;imshow(PR_ENER,[],'border','tight');title('�����ֲ�����');
imcontrast();
end

%%
% �Ľ׸���������ɢ�㷨
function Denoised_Out = Ada_ForthOrder_PR(I_ori, I_noi, I_Fide, PR_ENER,Dt, Item, k, Alpha, Lambada)
%%ImΪ��������ͼ��DtΪʱ�䲽����ItemΪ����������KcΪ��ɢ��������ɢ�LambadaΪ��ɢ�����еı�����Ĵ�С
niter = Item;
del_t = Dt;
[M,N]=size(I_noi);
SSIM_pro = 0;
Denoised_Out = zeros(N,N);
%%%%���������Ľ�ƫ΢�ַ���
for iter = 1:niter
    % ��1��������ͼ���ڷ��ߣ���ֱ��Ե�������ߣ����ű�Ե���򣩵Ķ���ƫ����
    ds =1;
    diff_1 =padarray(I_noi,[ds,ds],'symmetric','both');  %ͼ���Ե����
    % North, South, East and West pixel
    deltaN = diff_1(1:M,  2:N+1);
    deltaS = diff_1(3:M+2,2:N+1);
    deltaE = diff_1(2:M+1,3:N+2);
    deltaW = diff_1(2:M+1,  1:N);
    
    deltaNW = diff_1(1:M,   1:N);
    deltaSE = diff_1(3:M+2, 3:N+2);
    deltaNE = diff_1(1:M,   3:N+2);
    deltaSW = diff_1(3:M+2, 1:N);
    
    Ux = (deltaN-deltaS)./2;
    Uy = (deltaW-deltaE)./2;
    Uxx = deltaN + deltaS - 2.*I_noi;
    Uyy = deltaW + deltaE - 2.*I_noi;
    Uxy= (deltaNW + deltaSE - deltaNE -deltaSW)./4;   %% ����һ�׶���ƫ΢������
    
    Ux_2 = Ux.^2;
    Uy_2 = Uy.^2;
    Sum_xy2 = Ux_2 + Uy_2;
    Tidu_Mo = Sum_xy2.^0.5;
    
    min_p = min(PR_ENER(:));
    max_p = max(PR_ENER(:));
    max_T = max(Tidu_Mo(:));
    PR_ENER = (PR_ENER-min_p) / (max_p-min_p) *max_T;    %% ��һ���ֲ�����
    
    Dnn = (Uxx.*Ux_2 + 2.*Ux.*Uy.*Uxy + Uyy.*Uy_2)./(Sum_xy2 + eps);   %% ��ʽ14
    Dtt = (Uxx.*Uy_2 - 2.*Ux.*Uy.*Uxy + Uyy.*Ux_2)./(Sum_xy2 + eps);   %%
    
    % ��2�������� c(|deltaU|) ��ɢ������ʽ5
    Cs = k^2./(k^2 + (Tidu_Mo + Alpha * PR_ENER).^2);   %% ��ʽ��5���Ľ��
    %������������ Dxx_yy��������˹����
    P_Dxx_yy = Cs.^2.*Dnn + Cs.*Dtt;          %%% �Դ���������˹����
    %    P_Dxx_yy = Cs.*Dnn + Cs.*Dtt;          %%% �Դ���������˹����
    diff_2 = padarray(P_Dxx_yy,[ds,ds],'symmetric','both');  %%�����Ե
    
    %  North, South, East and West pixel
    g_deltaN = diff_2(1:M,  2:N+1);
    g_deltaS = diff_2(3:M+2,2:N+1);
    g_deltaE = diff_2(2:M+1,3:N+2);
    g_deltaW = diff_2(2:M+1,  1:N);
    L_Dxx_yy = (g_deltaN + g_deltaS + g_deltaE + g_deltaW) - 4 * P_Dxx_yy;
    
    % �����㴦����
    %    I_noi = I_noi -del_t*L_Dxx_yy;
    I_noi = I_noi -del_t*L_Dxx_yy - del_t * Lambada*(I_noi - I_Fide);  % �ӱ�����
    
    SSIM = mssim(I_noi, I_ori);
    if(SSIM>SSIM_pro)
        SSIM_pro = SSIM;
        Denoised_Out = I_noi;
    else
        fprintf('ģ�͵���������%d\n',iter);
        break;
    end
end

end