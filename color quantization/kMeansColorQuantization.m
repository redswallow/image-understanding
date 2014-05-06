function O = kMeansColorQuantization(I, k)
    [m,n,channel] = size(I);
    I = reshape(I, m*n, channel);
    I = double(I);
    [IDX, C] = kmeans(I, k);
    palette = round(C);
    IDX = uint8(IDX);
    IDX = reshape(IDX, [m,n]);
    O = zeros(m,n,channel);
    for i = 1 : m
        for j = 1 : n
            O(i,j,:) = palette(IDX(i,j),:);
        end
    end
    O = uint8(O);
end