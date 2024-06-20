function []=fun_clustering_plot(a01,a02,a03,fig_nmb)
%
%

a01=1.05*a01/max(abs(a01(:)));
a02=1.05*a02/max(abs(a02(:)));
a03=1.05*a03/max(abs(a03(:)));

ms=8;
A=[[0;1] [-sqrt(3)/2;-1/2] [sqrt(3)/2;-1/2]];

figure(fig_nmb)
hold off

plot(0,0,'ko','MarkerFaceColor','w','markersize',5)
hold on

plot([A(1,:),0],[A(2,:),1],'--','color',.7*[1, 1, 1]);

nmb_of_cluster_label=length(a01(1,:));

for jj=1:nmb_of_cluster_label
    plot(a01(1,jj),a01(2,jj),'mo','markerfacecolor','m','markersize',ms)
    hold on
end

for jj=1:nmb_of_cluster_label
    plot(a02(1,jj),a02(2,jj),'co','markerfacecolor','c','markersize',ms)
end

for jj=1:nmb_of_cluster_label
    plot(a03(1,jj),a03(2,jj),'go','markerfacecolor','g','markersize',ms)
end

plot(0,0,'ko','MarkerFaceColor','w','markersize',5)
plot(0,1,'ko','MarkerFaceColor','w','markersize',5)
plot(-sqrt(3)/2,-1/2,'ko','MarkerFaceColor','w','markersize',5)
plot(sqrt(3)/2,-1/2,'ko','MarkerFaceColor','w','markersize',5)

text(0, 1.1, {'"Digit 1"'},'color','m','FontSize',14)
text(-1, -.6, {'"Digit 2"'},'color','b','FontSize',14)
text(.7, -.6, {'"Digit 3"'},'color','g','FontSize',14)

axis([-1.1 1.1 -.75 1.2])

end