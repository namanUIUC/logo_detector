function pairs=matchFeatures(testFeatures,logoFeatures,dthr)

% Calculate all Euclidia distances between all features
Distances = pdist2(logoFeatures.descriptors', testFeatures.descriptors');

% Find min distances and sort them
[minDists,id_logo]=min(Distances);
[minDists,id_im]=sort(minDists,'ascend');
id_logo=id_logo(id_im);

% Reject distances higher than dthr
siz=length(find(minDists<dthr));
id_logo=id_logo(1:siz);
id_im=id_im(1:siz);

% Construct the pars structure
pairs.id_logo = id_logo;
pairs.id_im = id_im;
