function correspondance=FGCT(testFeatures,logoFeatures,pairs,alpha,sigma)

% Break if number of pairs are unsuficient
if length(pairs.id_logo)<6
    correspondance=0;
    return
end

% Construct matrix D_L
DL = pdist2(logoFeatures.frames(1:2,pairs.id_logo)',logoFeatures.frames(1:2,pairs.id_logo)');

% Construct matrix D_I
DI = pdist2(testFeatures.frames(1:2,pairs.id_im)',testFeatures.frames(1:2,pairs.id_im)');
SI = repmat(testFeatures.frames(3,pairs.id_im)',1,size(DI,2));
SL = repmat(logoFeatures.frames(3,pairs.id_logo)',1,size(DL,2));
SclM = SL./SI;
DI = DI.*SclM;

%Construct matrix D
Dab = exp(-(DL-DI).^2./(sigma*SclM).^2);
D = Dab+Dab';

% Initialize similarity vector K
K = ones(size(Dab,1),1)/size(Dab,1);
dK = K;

%Iterative procedure to calculate the fixed point K
while (norm(dK,2)>10^-5)
    G = exp(alpha*D*K);
    dK = G/norm(G,1)-K;
    K = K+dK;
end

% Calculate Correspondance
[Ks,idx]=sort(K,'descend');
Q=length(K);
Ks=repmat(Ks,1,Q) .* triu(ones(Q),0);
Dtemp=D(idx,idx);
normVector=Q./[1:Q]';
consistency=diag(Ks'*Dtemp*Ks).*normVector;
correspondance=sum(consistency)/logoFeatures.numFeatures;
