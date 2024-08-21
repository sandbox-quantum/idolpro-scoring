import os

import torch
from rdkit import Chem

from ocpmodels.models import PaiNN

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCPDataDummy:
    """
    Wrapper for data going into OCP model
    """

    def __init__(
        self, atomic_numbers=None, coordinates=None, indices=None, natoms=None
    ):
        self.atomic_numbers = atomic_numbers
        self.pos = coordinates
        self.batch = indices
        self.natoms = natoms


class TorchSAScore:
    """
    Model for approximating SA score.
    """

    def __init__(self):
        params = {
            "num_atoms": None,
            "bond_feat_dim": None,
            "num_targets": None,
            "hidden_channels": 256,
            "num_layers": 4,
            "num_rbf": 64,
            "cutoff": 8.0,
            "max_neighbors": 30,
            "regress_forces": False,
            "use_pbc": False,
            "num_elements": 10,
        }

        model = PaiNN(**params)
        model.load_state_dict(torch.load(os.path.join(cwd, "scoring", "torchsa.ckpt")))
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model
        self.conv_list = [6, 7, 8, 16, 5, 35, 17, 15, 53, 9]

    def score(self, ligands: list[Chem.Mol]) -> torch.tensor:
        """
        Score ligands according to torchSA
        Args:
            ligands: ligands to score
        Returns:
            tensor of SA scores for each ligand
        """
        ans_all, crs_all, ids_all, natoms_all = [], [], [], []
        for i, ligand in enumerate(ligands):

            crs = (
                torch.from_numpy(ligand.GetConformer().GetPositions())
                .float()
                .to(device)
            )
            ans = [
                ligand.GetAtomWithIdx(i).GetAtomicNum()
                for i in range(ligand.GetNumAtoms())
            ]
            converted_ans = torch.tensor(
                [self.conv_list.index(a) for a in ans], device=device
            )
            atomic_probs = torch.nn.functional.one_hot(
                converted_ans, num_classes=10
            ).float()
            ids = torch.full(
                size=(len(ans),), fill_value=i, device=device, dtype=torch.long
            )

            ans_all.append(atomic_probs)
            crs_all.append(crs)
            ids_all.append(ids)
            natoms_all.append(len(ans))

        batch = OCPDataDummy(
            atomic_numbers=torch.cat(ans_all, dim=0),
            coordinates=torch.cat(crs_all, dim=0),
            indices=torch.cat(ids_all, dim=0),
            natoms=torch.tensor(natoms_all),
        )

        return torch.clamp(self.model(batch), min=1, max=10)
