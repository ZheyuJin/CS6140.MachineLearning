series = 5 * (1:length(activeError));

plot(series ,activeError',series ,randError');
legend('active','rand');
ylim([0 1]);