function Psnr = Cal_Psnr(I_ori, I_de)
%%I_oriΪԭʼ�ο�ͼ��I_deΪȥ���ͼ��
%����ָ�ͼ���MSE
[M, N] = size(I_ori);
u1=I_ori-I_de;
u1=u1.*u1;
u1=sum(u1(:));
MSE=u1/(M*N);
%sprintf('ƽ���������Ϊ:%f',MSE);
   
%����ͼ��ķ�ֵ�����
p=(255*255)/(MSE+eps);
Psnr=10*log10(p);
%sprintf('��ֵ�����Ϊ:%f',Psnr);
end