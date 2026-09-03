import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



# ============================================================
# InfoNCE estimator for I(X;Z)
# ============================================================

class InfoNCE(nn.Module):

    def __init__(
            self,
            dim_x,
            dim_z,
            proj_dim=256,
            temperature=0.07
    ):
        super().__init__()

        self.temperature = temperature


        self.encoder_x = nn.Sequential(

            nn.Linear(
                dim_x,
                proj_dim
            ),

            nn.ReLU(),

            nn.Linear(
                proj_dim,
                proj_dim
            )
        )


        self.encoder_z = nn.Sequential(

            nn.Linear(
                dim_z,
                proj_dim
            ),

            nn.ReLU(),

            nn.Linear(
                proj_dim,
                proj_dim
            )
        )



    def forward(
            self,
            x,
            z
    ):

        x = self.encoder_x(x)

        z = self.encoder_z(z)



        x = F.normalize(
            x,
            dim=1
        )

        z = F.normalize(
            z,
            dim=1
        )



        logits = torch.matmul(
            x,
            z.T
        )


        logits /= self.temperature



        labels = torch.arange(
            x.size(0),
            device=x.device
        )


        loss = F.cross_entropy(
            logits,
            labels
        )


        return loss



# ============================================================
# Estimate I(X;Z)
# ============================================================

def estimate_I_XZ(
        X,
        Z,
        epochs=100,
        lr=1e-4,
        batch_size=256
):


    device=X.device


    estimator=InfoNCE(
        X.shape[1],
        Z.shape[1]
    ).to(device)



    optimizer=torch.optim.Adam(
        estimator.parameters(),
        lr=lr
    )


    N=X.size(0)



    estimator.train()


    for epoch in range(epochs):


        idx=torch.randperm(
            N,
            device=device
        )


        for i in range(
                0,
                N,
                batch_size
        ):


            batch_idx=idx[
                i:i+batch_size
            ]


            x=X[
                batch_idx
            ]

            z=Z[
                batch_idx
            ]


            loss=estimator(
                x,
                z
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



    estimator.eval()


    total_loss=0

    count=0


    with torch.no_grad():

        for i in range(
                0,
                N,
                batch_size
        ):


            x=X[
                i:i+batch_size
            ]

            z=Z[
                i:i+batch_size
            ]


            if x.size(0)<2:
                continue


            loss=estimator(
                x,
                z
            )


            total_loss += loss.item()

            count+=1



    loss=total_loss/(count+1e-8)



    mi=torch.log(
        torch.tensor(
            N,
            dtype=torch.float,
            device=device
        )
    )-loss



    return mi.item()



# ============================================================
# Estimate I(Z;Y)
# H(Y)-H(Y|Z)
# ============================================================


class LabelPredictor(nn.Module):

    def __init__(
            self,
            dim_z,
            num_classes
    ):
        super().__init__()


        self.fc=nn.Linear(
            dim_z,
            num_classes
        )


    def forward(self,z):

        return self.fc(z)



def estimate_I_ZY(
        Z,
        Y,
        epochs=100,
        lr=1e-3
):


    device=Z.device


    num_classes=int(
        Y.max()+1
    )


    predictor=LabelPredictor(
        Z.shape[1],
        num_classes
    ).to(device)



    optimizer=torch.optim.Adam(
        predictor.parameters(),
        lr=lr
    )


    dataset=torch.utils.data.TensorDataset(
        Z,
        Y
    )


    loader=torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True
    )



    predictor.train()


    for epoch in range(epochs):


        for z,y in loader:


            logits=predictor(
                z
            )


            loss=F.cross_entropy(
                logits,
                y
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



    predictor.eval()


    with torch.no_grad():


        logits=predictor(
            Z
        )


        ce=F.cross_entropy(
            logits,
            Y
        ).item()



    # empirical entropy H(Y)

    prob=torch.bincount(
        Y
    ).float()


    prob/=prob.sum()


    entropy=-(

        prob *
        torch.log(
            prob+1e-8
        )

    ).sum().item()



    I=entropy-ce


    return max(I,0)



# ============================================================
# Extract timm ResNet50 feature
# ============================================================


@torch.no_grad()
def extract_features(
        model,
        loader,
        device
):


    model.eval()


    X=[]

    Z=[]

    Y=[]


    correct=0

    total=0



    for images,labels in tqdm(loader):


        images=images.to(device)

        labels=labels.to(device)



        # fixed input representation
        # avoid changing X during training

        x=F.adaptive_avg_pool2d(
            images,
            1
        )


        x=x.flatten(
            1
        )



        z=model.forward_features(
            images
        )


        if z.dim()==4:


            z=F.adaptive_avg_pool2d(
                z,
                1
            )

            z=z.flatten(
                1
            )



        logits=model(
            images
        )


        pred=logits.argmax(
            dim=1
        )


        correct+=(
            pred==labels
        ).sum().item()


        total+=labels.size(0)



        X.append(
            x
        )


        Z.append(
            z
        )


        Y.append(
            labels
        )



    X=torch.cat(X)

    Z=torch.cat(Z)

    Y=torch.cat(Y)



    acc=100*correct/total



    return X,Z,Y,acc



# ============================================================
# Main interface
# ============================================================


def evaluate_representation(
        model,
        test_loader,
        device,
        mine_epochs=100
):


    print(
        "\n========== Information Analysis =========="
    )


    X,Z,Y,acc=extract_features(
        model,
        test_loader,
        device
    )


    print(
        f"Accuracy : {acc:.2f}%"
    )



    X=F.normalize(
        X,
        dim=1
    )


    Z=F.normalize(
        Z,
        dim=1
    )



    print(
        "Estimating I(Z;Y)"
    )


    I_ZY=estimate_I_ZY(
        Z,
        Y,
        epochs=mine_epochs
    )



    print(
        "Estimating I(X;Z)"
    )


    I_XZ=estimate_I_XZ(
        X,
        Z,
        epochs=mine_epochs
    )



    extra=I_XZ-I_ZY



    print("--------------------------------")

    print(
        f"I(Z;Y)              : {I_ZY:.4f}"
    )


    print(
        f"I(X;Z)              : {I_XZ:.4f}"
    )


    print(
        f"I(X;Z)-I(Z;Y)       : {extra:.4f}"
    )

    print("--------------------------------")



    return {

        "Accuracy":acc,

        "I(Z;Y)":I_ZY,

        "I(X;Z)":I_XZ,

        "Extra_Info":extra
    }