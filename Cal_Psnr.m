function Psnr = Cal_Psnr(I_ori, I_de)
%%I_ori为原始参考图像，I_de为去噪后图像
%计算恢复图像的MSE
[M, N] = size(I_ori);
u1=I_ori-I_de;
u1=u1.*u1;
u1=sum(u1(:));
MSE=u1/(M*N);
%sprintf('平均均方误差为:%f',MSE);
   
%计算图像的峰值信噪比
p=(255*255)/(MSE+eps);
Psnr=10*log10(p);
%sprintf('峰值信噪比为:%f',Psnr);
end