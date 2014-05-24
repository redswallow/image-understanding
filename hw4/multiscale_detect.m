function [x, y, score, scale] = multiscale_detect(I,template,ndet)
    step = 1;
    while (size(template,1)<size(I,1) && size(template,1)<size(I,2))
        [pyrmaidx(step, :), pyrmaidy(step, :), pyrmaidscore(step, :)] = detect(I, template, ndet);
        pyrmaidscale(step, :) = ones(size(pyrmaidscore(step, :)))*((1/0.8).^(step-1));
        pyrmaidscore(step, :) = pyrmaidscore(step, :);
        I = imresize(I,0.8);
        step = step + 1;
    end
    
    pyrmaidx = reshape(pyrmaidx,1,[]);
    pyrmaidy = reshape(pyrmaidy,1,[]);
    pyrmaidscore = reshape(pyrmaidscore,1,[]);
    pyrmaidscale = reshape(pyrmaidscale,1,[]);
    
    % sort response from high to low
    [val,ind] = sort(pyrmaidscore(:),'descend');
    
    % work down the list of responses, removing overlapping detections as we go
    i = 1;
    detcount = 0;

    while ((detcount < ndet) && (i < length(ind)))
        % now convert yblock,xblock to pixel coordinates 
        ypixel = round(pyrmaidy(ind(i)) * pyrmaidscale(ind(i)));
        xpixel = round(pyrmaidx(ind(i)) * pyrmaidscale(ind(i)));

        % check if this detection overlaps any detections which we've already added to the list
        if (detcount == 0)
            overlap = 0;
        else
            dist = (y-ypixel).^2 + (x-xpixel).^2;
            overlap = sum(dist < (16*8*pyrmaidscale(ind(i))).^2);
        end
  
        % if not, then add this detection location and score to the list we return
        if (~overlap)
            detcount = detcount+1;
            x(detcount) = xpixel;
            y(detcount) = ypixel;
            score(detcount) = pyrmaidscore(ind(i));
            scale(detcount) = pyrmaidscale(ind(i));
        end
        i = i + 1;
    end
end