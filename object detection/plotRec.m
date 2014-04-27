function plotRec(X, Y)
    Xsquare = [X(1),X(2),X(2),X(1),X(1)];
    Ysquare = [Y(1),Y(1),Y(2),Y(2),Y(1)];
    plot(Xsquare, Ysquare, 'g', 'linewidth', 2);
end