function S = drawskt3D(path,a,s,e,conf)

B=[];

file=sprintf([path,'a%02i_s%02i_e%02i_skeleton3D.txt'],a,s,e);
fp=fopen(file);
if (fp>0)
   A=fscanf(fp,'%f');
   B=[B; A];
   fclose(fp);
end

l=size(B,1)/4;
B=reshape(B,4,l);
if (~conf)
    B = B(1:3,:);
    S=reshape(B,20*3,l/20);
else
    S=reshape(B,20*4,l/20);
end