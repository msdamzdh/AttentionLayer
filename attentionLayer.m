classdef attentionLayer < nnet.layer.Layer  ...
        & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        Nhead
        InFormat
        UseMask
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
        Wq
        Wk
        Wv
        Wo
    end

    methods
        function layer = attentionLayer(InputDim,QueryDim,ValuDim,...
                OutputDim,NumberOfHead,InputFormat,UseMask,Name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            arguments
                InputDim {mustBeInteger,mustBeGreaterThan(InputDim,0)}
                QueryDim {mustBeInteger,mustBeGreaterThan(QueryDim,0)}
                ValuDim {mustBeInteger,mustBeGreaterThan(ValuDim,0)}
                OutputDim {mustBeInteger,mustBeGreaterThan(OutputDim,0)}
                NumberOfHead {mustBeInteger,mustBeGreaterThan(NumberOfHead,0)}
                InputFormat {mustBeMember(InputFormat,["CBT","CTB","BTC","BCT","TCB","TBC"])}
                UseMask = false;
                Name ="attentionLayer";
            end
            layer.Wq = dlarray(rand(InputDim,QueryDim));
            layer.Wk = dlarray(rand(InputDim,QueryDim));
            layer.Wv = dlarray(rand(InputDim,ValuDim));
            layer.Wo = dlarray(rand(ValuDim,OutputDim));
            layer.Nhead=NumberOfHead;
            layer.InFormat = InputFormat;
            if UseMask
                layer.UseMask = "causal";
            else
                layer.UseMask = "none";
            end
            layer.Name=Name;
        end

        function [Z] = predict(layer,X)

            X = stripdims(X);

            if layer.InFormat=="CBT"
                X=permute(X,[3,1,2]);
            elseif layer.InFormat=="CTB"
                X=permute(X,[2,1,3]);
            elseif layer.InFormat=="BTC"
                X=permute(X,[2,3,1]);
            elseif layer.InFormat=="BCT"
                X=permute(X,[3,2,1]);
            elseif layer.InFormat=="TBC"
                X=permute(X,[1,3,2]);
            end
            Q = pagemtimes(X,layer.Wq);
            K = pagemtimes(X,layer.Wk);
            V = pagemtimes(X,layer.Wv);
            A = attention(Q,K,V,layer.Nhead,DataFormat="TCB",AttentionMask=layer.UseMask);
            Z = dlarray(pagemtimes(A,layer.Wo),"TCB");
        end
    end
end