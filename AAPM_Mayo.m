%%主要参考文章：“An Anisotropic Fourth-Order Diffusion Filter for Image Noise Removal 2011 Mohammad”
%%文章：“An adaptive diffusion coefficient selection for image denoising 2017 Hossein”
%%文章：“Artifact Suppressed Nonlinear Diffusion Filtering for Low-Dose CT Image Processing 2019 YI LIU”
%%算法原理：利用获取的残差图像求得残差局部能量(残差局部方差)，结合图像梯度膜，重新定义扩散函数C(s),在抑制条形伪影的情况下，更好的保存图像细节和边缘;
%%结合四阶各向异性偏微分方程，在边缘，纹理处减少扩散，在平滑区域增强扩散的优势，进一步抑制条形伪影和保留图像边缘。
%% 此代码主要处理AAPM-Mayo-CT-Challenge模拟低剂量CT图像
function main
close all
clear all
clc

addpath(genpath(pwd))

%L014 腹腔图像
% fileName_Nonoise = '.\pic\Abdomen\L014_Full_38.png';
% fileName_Noise = '.\pic\Abdomen\L014_Quarter_38.png';

%L006 胸腔图像
% fileName_Nonoise = '.\pic\Chest\L019_Full_8.png';
% fileName_Noise = '.\pic\Chest\L019_Quarter_8.png';

%L006 骨腔图像
fileName_Nonoise = '.\pic\Pelvic\L006_Full_66.png';
fileName_Noise = '.\pic\Pelvic\L006_Quarter_66.png';

g= double(rgb2gray(imread(fileName_Nonoise)));                        % 无噪声图像 AAPM-Mayo-CT-Challenge低剂量CT图像
I_noi = double(rgb2gray(imread(fileName_Noise)));                     % 噪声图像

figure,imshow(I_noi,[],'border','tight');title('低剂量CT图像');
imcontrast;
figure,imshow(g,[],'border','tight');title('标准剂量CT图像');
imcontrast;

% 定量评价指标
MSSIM_a = mssim(g, I_noi);
fprintf('结构相似度为：%.4f\n',MSSIM_a);
Psnr_a = Cal_Psnr(g, I_noi);
fprintf('峰值信噪比为：%.4f\n',Psnr_a);
[FSIMa, FSIMca ] = FeatureSIM (g, I_noi);
fprintf('特征相似度为：%.4f\n',FSIMa);
gmsda = GMSD(g, I_noi)   ;
fprintf('幅度相似性偏差为：%.4f\n',gmsda);                  %GMSD 梯度幅度相似性偏差 （值越小越好）

I_Fide = I_noi;

%扩散阈值的计算
[Gx,Gy]=gradient(I_noi);
gradientimage = sqrt(Gx .* Gx + Gy .* Gy);
gradientimage1=gradientimage(:);
ind= find(gradientimage1>0);
kk=gradientimage1(ind);
kk=kk(:);
len=size(kk,1);
kc=sum(kk)/len*0.9;

% 参数设置
niter = 20000;   %四阶偏微分方程的迭代次数，根据ssim决定具体多少次迭代
del_t = 0.03;    %时间步长
beta = 1.8;      % 噪声方差系数

% 腹腔图像
% Alpha = 2.0;     % 残差能量权重系数
% Lambada = 1;  % 保真项权重系数  0.6

% 胸腔图像
% Alpha = 4.6;     % 残差能量权重系数
% Lambada = 0.6;  % 保真项权重系数

% 盆腔图像
Alpha = 2.2;     % 残差能量权重系数
Lambada = 0.5;  % 保真项权重系数


%%求图像的残差
% sub = I_noi - g;
% sigma = var(sub(:));      %按原论文的意思为高斯白噪声的噪声标准差，低剂量CT图像为条形伪影和斑点噪声
% sigma_sq = sigma^2;

I_noi = gpuArray(I_noi);  %GPU加速
g = gpuArray(g);  %GPU加速

tic
% 残差图像和残差局部能量获取
[PR_ENER] = Hoss_PR(I_noi,kc,beta);
%%各项异性四阶偏微分去噪
I_Out = Ada_ForthOrder_PR(g, I_noi, I_Fide, PR_ENER, del_t, niter, kc, Alpha, Lambada);
toc

% 将数据从GPU放到CPU
g=gather(g);
I_out = gather(I_Out);
I_noi = gather(I_noi);

% 定量评价指标
MSSIM_a = mssim(g, I_out);
fprintf('结构相似度为：%.4f\n',MSSIM_a);
Psnr_a = Cal_Psnr(g, I_out);
fprintf('峰值信噪比为：%.4f\n',Psnr_a);
[FSIMa, FSIMca ] = FeatureSIM (g, I_out);
fprintf('特征相似度为：%.4f\n',FSIMa);
gmsda = GMSD(g, I_out)   ;
fprintf('幅度相似性偏差为：%.4f\n',gmsda);                  %GMSD 梯度幅度相似性偏差 （值越小越好）

figure,imshow(I_out,[],'border','tight');
imcontrast;

end

%计算图像残差信息
function [PR_ENER] = Hoss_PR(I_noi,Kc,beta)
%%输入：I_ori为无噪声图像；I_noi为输入噪声图像；Kc为扩散函数中扩散项；theta_sq为图像噪声方差；beta为常数；控制残差中的信息多少
%%输出：PR_Im为噪声图像与各向异性四阶扩散后的差值图像；PR_ENER为残差局部能量；theta_sq_nor为残差局部方差
Gauss_He_n=fspecial('gaussian',5,5);
Assump =imfilter(I_noi, Gauss_He_n);
sub_Ass = I_noi-Assump;
theta_sq = var(sub_Ass(:));
thread_var = beta * theta_sq;
Var_Ir = 0; Time_Forth = 0;
ds = 1; % 图像的扩展边缘
kc = Kc; % 各向异性四阶扩散的阈值
del_t = 0.03; %四阶各向异性迭代步长
I_Noi_re = I_noi;
[M,N] = size(I_noi);
%%第一步，先进行四阶各向异性扩散，求取图像的差
while(Var_Ir < thread_var && Time_Forth<20000)
    
    % 第1步，计算图像在法线（垂直边缘）和切线（沿着边缘方向）的二阶偏导数
    diff_1 =padarray(I_noi,[ds,ds],'symmetric','both');  %图像边缘处理
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
    Uxy= (deltaNW + deltaSE - deltaNE -deltaSW)./4;   %% 计算一阶二阶偏微分算子
    
    Ux_2 = Ux.^2;
    Uy_2 = Uy.^2;
    Sum_xy2 = Ux_2 + Uy_2;
    Tidu_Mo = Sum_xy2.^0.5;
    Dnn = (Uxx.*Ux_2 + 2.*Ux.*Uy.*Uxy + Uyy.*Uy_2)./(Sum_xy2 + eps);   %% 4阶各向异性扩散文章公式14
    Dtt = (Uxx.*Uy_2 - 2.*Ux.*Uy.*Uxy + Uyy.*Ux_2)./(Sum_xy2 + eps);   %%
    
    % 第2步，计算 c(|deltaU|) 扩散函数公式5
    Cs = kc^2./(kc^2 + (Tidu_Mo).^2);   %% 公式（5）的结合
    
    %第3步，计算 Dxx_yy的拉普拉斯算子
    P_Dxx_yy = Cs.^2.*Dnn + Cs.*Dtt;              %%% 对此求拉普拉斯算子
    diff_2 = padarray(P_Dxx_yy,[ds,ds],'symmetric','both');  %%扩充边缘
    
    %  North, South, East and West pixel
    g_deltaN = diff_2(1:M,  2:N+1);
    g_deltaS = diff_2(3:M+2,2:N+1);
    g_deltaE = diff_2(2:M+1,3:N+2);
    g_deltaW = diff_2(2:M+1,  1:N);
    L_Dxx_yy = (g_deltaN + g_deltaS + g_deltaE + g_deltaW) - 4 * P_Dxx_yy;
    
    % 最后计算处理结果
    I_noi = I_noi -del_t*L_Dxx_yy;
    
    PR_Im = I_Noi_re - I_noi;
    mean_Ir=mean2(PR_Im);   % 均值
    
    R=(PR_Im-mean_Ir).^2;
    Var_Ir=sum(sum(R))/(M*N);
    
    Time_Forth = Time_Forth +1;
    %     sprintf('方差为：%f',Var_Ir)
    %
end
fprintf('求残差图像迭代次数为：%d\n',Time_Forth);
%figure;imshow(PR_Im,[]); title('四阶各向异性扩散残差图像');

%%
% 第二步，计算残差局部能量
filter_width=9;
PR_Im =padarray(PR_Im,[filter_width,filter_width],'symmetric','both');  %图像边缘处理

PR_Im =medfilt2(PR_Im,[3,3]);
mean_he=fspecial('average',[filter_width,filter_width]);%均值滤波
IR_mean=imfilter(PR_Im,mean_he); %对残差图像进行局部均值
P=(PR_Im-IR_mean).^2;

gaus_he=fspecial('gaussian',filter_width,5);
LocalPower =imfilter(P, gaus_he);
PR_ENER=LocalPower((1+filter_width):(M+filter_width),(1+filter_width):(N+filter_width));

figure;imshow(PR_ENER,[],'border','tight');title('残留局部能量');
imcontrast();
end

%%
% 四阶各向异性扩散算法
function Denoised_Out = Ada_ForthOrder_PR(I_ori, I_noi, I_Fide, PR_ENER,Dt, Item, k, Alpha, Lambada)
%%Im为输入噪声图像，Dt为时间步长，Item为迭代次数，Kc为扩散函数中扩散项；Lambada为扩散函数中的保真项的大小
niter = Item;
del_t = Dt;
[M,N]=size(I_noi);
SSIM_pro = 0;
Denoised_Out = zeros(N,N);
%%%%各向异性四阶偏微分方程
for iter = 1:niter
    % 第1步，计算图像在法线（垂直边缘）和切线（沿着边缘方向）的二阶偏导数
    ds =1;
    diff_1 =padarray(I_noi,[ds,ds],'symmetric','both');  %图像边缘处理
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
    Uxy= (deltaNW + deltaSE - deltaNE -deltaSW)./4;   %% 计算一阶二阶偏微分算子
    
    Ux_2 = Ux.^2;
    Uy_2 = Uy.^2;
    Sum_xy2 = Ux_2 + Uy_2;
    Tidu_Mo = Sum_xy2.^0.5;
    
    min_p = min(PR_ENER(:));
    max_p = max(PR_ENER(:));
    max_T = max(Tidu_Mo(:));
    PR_ENER = (PR_ENER-min_p) / (max_p-min_p) *max_T;    %% 归一化局部能量
    
    Dnn = (Uxx.*Ux_2 + 2.*Ux.*Uy.*Uxy + Uyy.*Uy_2)./(Sum_xy2 + eps);   %% 公式14
    Dtt = (Uxx.*Uy_2 - 2.*Ux.*Uy.*Uxy + Uyy.*Ux_2)./(Sum_xy2 + eps);   %%
    
    % 第2步，计算 c(|deltaU|) 扩散函数公式5
    Cs = k^2./(k^2 + (Tidu_Mo + Alpha * PR_ENER).^2);   %% 公式（5）的结合
    %第三步，计算 Dxx_yy的拉普拉斯算子
    P_Dxx_yy = Cs.^2.*Dnn + Cs.*Dtt;          %%% 对此求拉普拉斯算子
    %    P_Dxx_yy = Cs.*Dnn + Cs.*Dtt;          %%% 对此求拉普拉斯算子
    diff_2 = padarray(P_Dxx_yy,[ds,ds],'symmetric','both');  %%扩充边缘
    
    %  North, South, East and West pixel
    g_deltaN = diff_2(1:M,  2:N+1);
    g_deltaS = diff_2(3:M+2,2:N+1);
    g_deltaE = diff_2(2:M+1,3:N+2);
    g_deltaW = diff_2(2:M+1,  1:N);
    L_Dxx_yy = (g_deltaN + g_deltaS + g_deltaE + g_deltaW) - 4 * P_Dxx_yy;
    
    % 最后计算处理结果
    %    I_noi = I_noi -del_t*L_Dxx_yy;
    I_noi = I_noi -del_t*L_Dxx_yy - del_t * Lambada*(I_noi - I_Fide);  % 加保真项
    
    SSIM = mssim(I_noi, I_ori);
    if(SSIM>SSIM_pro)
        SSIM_pro = SSIM;
        Denoised_Out = I_noi;
    else
        fprintf('模型迭代次数：%d\n',iter);
        break;
    end
end

end