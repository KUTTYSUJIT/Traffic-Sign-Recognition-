% % Function called within Traffic_Recognition.m for image-resizing
function res = imgresize(path, srcFiles)
    for i = 1 : length(srcFiles) 
    filename = strcat(path,srcFiles(i).name); 
    im = imread(filename); 
    k=imresize(im,[224,224]); 
    newfilename=strcat(path,srcFiles(i).name); 
    imwrite(k,newfilename,'png'); 
    end
end
