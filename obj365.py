from utils.general import Path, check_requirements, download, np, xyxy2xywhn
check_requirements(('pycocotools>=2.0', ))
from pycocotools.coco import COCO
from tqdm import tqdm

if __name__ == '__main__':
    path = '/media/yy/Test/yolonew/yolov5/data/Objects365/'
    coco = COCO(path + 'Annotations/val/val.json')
    labels = path + 'labels/val/'
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
    # out = ''
    # for a in names:
    #     out += '\'' + a + '\'' + ','
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds),
                       desc=f'Class {cid + 1}/{len(names)} {cat}'):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])  # image filename
            try:
                print(path.with_suffix('.txt').name)
                with open(labels + path.with_suffix('.txt').name, 'a') as file:
                    annIds = coco.getAnnIds(imgIds=im["id"],
                                            catIds=catIds,
                                            iscrowd=None)
                    for a in coco.loadAnns(annIds):
                        x, y, w, h = a[
                            'bbox']  # bounding box in xywh (xy top-left corner)
                        xyxy = np.array([x, y, x + w,
                                         y + h])[None]  # pixels(1,4)
                        x, y, w, h = xyxy2xywhn(
                            xyxy, w=width, h=height,
                            clip=True)[0]  # normalized and clipped
                        file.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
            except Exception as e:
                print(e)