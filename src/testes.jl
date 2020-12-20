##
OSBORNEA

CERI651CLS
DEVGLA1
GAUSS2LS
NELSONLS
OSBORNEA

##
S = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20]
##
m = 8
for i = 0:m
    print(S[:,mod(i,m)+1])
end
##
S