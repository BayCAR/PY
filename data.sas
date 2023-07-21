
libname cal 'C:\Users\shpry\PycharmProjects\cal';
run;

data data;
set cal.data;
x2=x*x;
if col>300;
run;

data data0;
set data;
col=251;
run;

data data2;
set data data0;run;

proc mixed;
class col;
model y=x x2 col /s outp=out;
run;


data out0;
set out;
if col=251;
ref=3632.75-0.07963x*-x2*0.00003;
run;
proc freq; table col;run;

proc gplot data=out0;
plot pred*x;
run;
