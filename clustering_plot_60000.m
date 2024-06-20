% ms=3;

mx=max([max(abs(a01(:))),max(abs(a02(:))),max(abs(a03(:)))]);
a01=1.05*a01/mx;
a02=1.05*a02/mx;
a03=1.05*a03/mx;

plot(0,0,'ko','MarkerFaceColor','w','markersize',5)
hold on

plot([A(1,:),0],[A(2,:),1],'--','color',.7*[1, 1, 1]);

for jj=1:length(a01(1,:))
    plot(a01(1,jj),a01(2,jj),'mo','markerfacecolor','m','markersize',ms)
    hold on
end

for jj=1:length(a02(1,:))
    plot(a02(1,jj),a02(2,jj),'co','markerfacecolor','c','markersize',ms)
end

for jj=1:length(a03(1,:))
    plot(a03(1,jj),a03(2,jj),'go','markerfacecolor','g','markersize',ms)
end

plot(0,0,'ko','MarkerFaceColor','w','markersize',5)
plot(0,1,'ko','MarkerFaceColor','w','markersize',5)
plot(-sqrt(3)/2,-1/2,'ko','MarkerFaceColor','w','markersize',5)
plot(sqrt(3)/2,-1/2,'ko','MarkerFaceColor','w','markersize',5)

text(0, 1.1, {'"Digit 1"'},'color',[.8, 0, .7],'FontSize',14)
text(-1, -.6, {'"Digit 2"'},'color','b','FontSize',14)
text(.5, -.6, {'"Digit 3"'},'color',[.5, .7, 0],'FontSize',14)

axis([-1.1 1.1 -.8 1.2])
% axis([-1.1 1.1 -1.2 1.2])