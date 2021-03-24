from models.siamese import SiameseRPNpp
import torch



if __name__=='__main__':
    target_test = torch.randn(1, 3, 127, 127)
    search_test = torch.randn(1, 3, 255, 255)

    siamese = SiameseRPNpp(anchor=1)

    box_out, cls_out = siamese(target_test, search_test)

    print(f'box size {box_out.size()}, classifier size : {cls_out.size()}')
