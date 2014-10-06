%update ANN weights
function [nw] = update_weights(rate,w, curDeriv, prevLayerOutput)
% not using momentum
nw= w + rate *  prevLayerOutput *curDeriv';

%using momentem
% momenteum = rate*0.0001;
% nw =w;
% nw= nw + rate *  prevLayerOutput *curDeriv';
% nw = nw + momenteum *w;


% [rows, cols] = size(w);
% for c=1:cols
%     for r=1:rows
%         nw(r,c)= nw(r,c) + rate *  prevLayerOutput(r) *curDeriv(r);
%     end
% end
end