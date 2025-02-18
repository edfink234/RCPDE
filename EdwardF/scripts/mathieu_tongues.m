%Plot APPROXIMATE Mathieu curves according to:
%Analytical expressions for stability regions in the Ince–Strutt diagram of Mathieu equation 
%Eugene I. Butikov, Am. J. Phys. 86, 257–267 (2018), https://doi.org/10.1119/1.5021895

%%%%%%%%%%%%%%%%%%%%%%%%%
cb=[0 0.4470 0.7410];
cr=[0.8500 0.3250 0.0980];
cg=[1 1 1]*0.8;
lw=2;
Npts=400;
L=-1;R=1.8;T=2; % Left, Right, Top
k1=linspace(L-0.5,0,Npts);
k2=linspace(L-0.5,0.25,Npts);
k3m=linspace(0.25,13/20,Npts);
k3p=linspace(13/20,0.25,Npts);
k4=linspace(13/20,1,Npts);
k5=linspace(1,R,Npts);
m1=2*sqrt((k1.*(k1-1).*(k1-4))./(3*k1-8));
m2=0.25*(sqrt((9-4*k2).*(13-20*k2))-(9-4*k2));
m3p=0.25*(9-4*k3p+sqrt((9-4*k3p).*(13-20*k3p)));
m3m=0.25*(9-4*k3m-sqrt((9-4*k3m).*(13-20*k3m)));
m4=sqrt((2*(k4-1).*(k4-4).*(k4-9))./(k4-5));
m5=2*sqrt((k5.*(k5-1).*(k5-4))./(3*k5-8));
k3=[k3m,k3p];
m3=[m3m,m3p];

%Cut to box:
D=T/10; % to make sure that we do not miss pts inside box
i1=find(k1>=L-D & k1<=R+D & m1<=T+D); m1=m1(i1);k1=k1(i1);
i2=find(k2>=L-D & k2<=R+D & m2<=T+D); m2=m2(i2);k2=k2(i2);
i3=find(k3>=L-D & k3<=R+D & m3<=T+D); m3=m3(i3);k3=k3(i3);
i4=find(k4>=L-D & k4<=R+D & m4<=T+D); m4=m4(i4);k4=k4(i4);
i5=find(k5>=L-D & k5<=R+D & m5<=T+D); m5=m5(i5);k5=k5(i5);

%areas to shade
x1=[k1 k1(1) k1(1)];
y1=[m1  0    m1(1)];
x2=[k2 k3];
y2=[m2 m3];
x3=[k4 k5];
y3=[m4 m5];

figure(2);clf
fill(x1,y1,cg,'EdgeColor',cg)
hold on
fill(x2,y2,cg,'EdgeColor',cg)
fill(x3,y3,cg,'EdgeColor',cg)
plot(k1,m1,'-','Color',cb,'LineWidth',lw)
plot(k2,m2,'-','Color',cb,'LineWidth',lw)
plot(k3,m3,'-','Color',cb,'LineWidth',lw)
plot(k4,m4,'-','Color',cb,'LineWidth',lw)
plot(k5,m5,'-','Color',cb,'LineWidth',lw)
hold off
axis([L R 0 T])

xlabel('$k$')
ylabel('$m$')

