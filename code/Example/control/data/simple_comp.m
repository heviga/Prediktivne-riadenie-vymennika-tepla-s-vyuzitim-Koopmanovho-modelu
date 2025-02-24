clear; close all;

y_koopman_noC = load('strejc_ref_x_3').y;
y_koopman_wC = load('koop_wC_ref_x_3').y;

u_koopman_noC = load('strejc_ref_u_3').u;
u_koopman_wC = load('koop_wC_ref_u_3').u;

part = [120:900];
dif = 49.4-46.902094;
%% plots
figure
hold on
plot(y_koopman_noC)
plot(y_koopman_wC)
stairs([0 120 420 720, 900], [46.902094+dif,  36.086426,  57.50674, 46.902094+dif,46.502094+dif])
legend('koopman_noC' , 'koopman_wC', "ref")
xlabel('time') 
ylabel('T [C]') 
%%
figure
hold on
plot(u_koopman_noC)
plot(u_koopman_wC)
legend('koopman_noC' , 'koopman_wC')
xlabel('time') 
ylabel('q [%]') 

%%
e_ko = sum(abs(y_koopman_noC(part)-48))
e_st = sum(abs(y_koopman_wC(part)-48))
kbetter = e_ko/e_st

%%
eu_ko = sum(abs(u_koopman_noC(part)-50))
eu_st = sum(abs(u_koopman_wC(part)-50))
kubetter = eu_ko/eu_st

%%
uabs_ko = sum(abs(u_koopman_noC(part)))
uabs_st = sum(abs(u_koopman_wC(part)))
kuabsbetter = uabs_ko/uabs_st
%%
uc_ko = sum((y_koopman_noC(part)-48).^2)*10+sum((u_koopman_noC(part)-50).^2)
uc_st = sum((y_koopman_wC(part)-48).^2)*10+sum((u_koopman_wC(part)-50).^2)
koopman_noC_better = uc_ko/uc_st

y_koopman_noC = load('koop_noC_ref_x').y;
y_koopman_wC = load('koop_wC_ref_x').y;

u_koopman_noC = load('koop_noC_ref_u').u;
u_koopman_wC = load('koop_wC_ref_u').u;

part = [60:900];
dif = 49.4-46.602094;

%% plots
figure
hold on
plot(y_koopman_noC)
plot(y_koopman_wC)
stairs([0 120 420 720, 900], [46.902094+dif,  36.086426,  57.50674, 46.902094+dif,46.502094+dif])
legend('koopman_noC' , 'koopman_wC', "ref")
xlabel('time') 
ylabel('T [C]') 
%%
figure
hold on
plot(u_koopman_noC)
plot(u_koopman_wC)
legend('koopman_noC' , 'koopman_wC')
xlabel('time') 
ylabel('q [%]') 

y_koopman_noC = load('koop_noC_ref_x_2').y;
y_koopman_wC = load('koop_wC_ref_x_2').y;

u_koopman_noC = load('koop_noC_ref_u_2').u;
u_koopman_wC = load('koop_wC_ref_u_2').u;

part = [60:900];
dif = 49.4-46.902094;
%% plots
figure
hold on
plot(y_koopman_noC)
plot(y_koopman_wC)
stairs([0 120 420 720, 900], [46.902094+dif,  36.086426,  57.50674, 46.902094+dif,46.502094+dif])
legend('koopman_noC' , 'koopman_wC', "ref")
xlabel('time') 
ylabel('T [C]') 
%%
figure
hold on
plot(u_koopman_noC)
plot(u_koopman_wC)
legend('koopman_noC' , 'koopman_wC')
xlabel('time') 
ylabel('q [%]') 
% %%
% 
% y_koopman_noC = load('koopman_noC_10_01_dis_p1_y_03').y;
% y_koopman_wC = load('koopman_wC_10_01_dis_p1_y_03').y;
% 
% u_koopman_noC = load('koopman_noC_10_01_dis_p1_u_03').u;
% u_koopman_wC = load('koopman_wC_10_01_dis_p1_u_03').u;
% 
% %% plots
% figure
% hold on
% plot(y_koopman_noC)
% plot(y_koopman_wC)
% 
% figure
% hold on
% plot(u_koopman_noC)
% plot(u_koopman_wC)
% 
% %%
% e_ko = sum(abs(y_koopman_noC(60:end)-52))
% e_st = sum(abs(y_koopman_wC(60:end)-52))
% kbetter = e_ko/e_st
% 
% %%
% eu_ko = sum(abs(u_koopman_noC(60:end)-50))
% eu_st = sum(abs(u_koopman_wC(60:end)-50))
% kubetter = eu_ko/eu_st
% 
% %%
% uabs_ko = sum(abs(u_koopman_noC(60:end)))
% uabs_st = sum(abs(u_koopman_wC(60:end)))
% kuabsbetter = uabs_ko/uabs_st
% %%
% uc_ko = sum((y_koopman_noC(60:end)-52).^2)*10+sum((u_koopman_noC(60:end)-50).^2)
% uc_st = sum((y_koopman_wC(60:end)-52).^2)*10+sum((u_koopman_wC(60:end)-50).^2)
% koopman_noC_better = uc_ko/uc_st
% 
