import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

TO_BE_IGNORED = ["README.txt"]

@DATASET_REGISTRY.register()
class BTXRD2(DatasetBase):

    dataset_dir = "BTXRD2"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
                ood = preprocessed["ood"]

        else:
            id_classnames = self.read_classnames(os.path.join(self.image_dir, "train"))
            ood_classnames = self.read_classnames(os.path.join(self.image_dir, "ood"))

            train = self.read_data(id_classnames, "train")
            test = self.read_data(id_classnames, "test")
            ood = self.read_data(ood_classnames, "ood")

            preprocessed = {"train": train, "test": test, "ood": ood}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            fewshot_file = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(fewshot_file):
                print(f"Loading preprocessed few-shot data from {fewshot_file}")
                with open(fewshot_file, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {fewshot_file}")
                with open(fewshot_file, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.id = test
        self.ood = ood

        super().__init__(train_x=train, val=test, test=test)


    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def read_classnames(self, image_dir):
        classnames = OrderedDict()
        crouse_classnames = os.listdir(image_dir)
        for i, classname in enumerate(crouse_classnames):
            classnames[classname] = classname.replace("_", " ")
        return classnames