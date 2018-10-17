function features=extractFeaturesBulk(directory)

dirList=dir(directory);
h = waitbar(0,'Extracting features');

j=0;
for i=1:length(dirList)
    if ~isempty(strfind(dirList(i).name, '.jpg')) || ~isempty(strfind(dirList(i).name, '.png'))
        j = j+1;
        dirList2(j) = dirList(i);
    end
end

% Instead can be used: parfor
for i=1:length(dirList2)
    features(i) = extractFeatures([directory dirList2(i).name]);
    waitbar(i/length(dirList2),h,['Extracting features from: ' directory])
end
close(h);pause(0.01)
