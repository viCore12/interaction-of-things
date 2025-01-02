# Human Object Interaction
## Phân tích yêu cầu
### Yêu cầu bài toán
- Phát hiện các hành động như: tương tác với nhân viên, tương tác với quảng cáo... dựa trên ảnh tĩnh.
### Hướng giải quyết
- Sử dụng model Human Object Interaction để phân loại các hành động dựa trên các cặp `<human, object>`.
- Có 2 dạng model HOI chính: **two-stage** và **one-stage**, trong đó các model **two-stage** sẽ bao gồm 2 mạng neural nối tiếp nhau: 1 mạng neural dùng để detect các human, object và 1 mạng dùng để phân loại các cặp human-object này. Trong khi đó các model **one-stage** sẽ thực hiện việc tìm ra các cặp human-object và phân loại interaction giữa chúng một cách trực tiếp.
- Trong bài toán này, ta sẽ sử dụng model dạng **two-stage** do ta có thể sử dụng các object đã detect được từ detector trong model, sau đó sử dụng các object này cho các module khác như tracking, counting... thay vì dùng một model độc lập. Hơn nữa việc sử dụng model two-stage cũng cấp cho ta khả năng debug dễ dàng hơn, ta có thể finetune model detector một cách dễ dàng thay vì phải train lại toàn bộ model HOI.
## Triển khai
### Yêu cầu:
- Đã test trên Ubuntu 24.04, Python 3.10.14.
### Dependencies
`
pip install -r requirements.txt
`

**Repo này đã được tùy chỉnh một số code so với repo UPT gốc, các code đã được thêm nằm ở dưới comment `Modified by Tuan` .**
## Các lệnh
### Training
```
cd upt

python main.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions train val --output-dir checkpoints/upt-r50-vcoco --print-interval 1 --epochs 300 --batch-size 8 --min-instances 1 --box-score-thresh 0.5 --lr-head 5e-8 --fg-iou-thresh 0.5 --detr_checkpoint.pth --human-idx 1 --num-classes 2
```
### Demo trên video
```
python main.py --human-idx 1 --video-path merged.mp4 --resume upt.pt --num-classes 2 --device cpu
```
## Tích hợp UPT với các module khác
Custom các phương thức khác trong class UPT để sử dụng các bounding box từ DETR cho các module khác nhau như tracking, counting. Sau đó lưu lại các thông tin cần thiết để truyền qua interaction head để vừa thực hiện tracking, counting... vừa phân loại được hành động mà không cần phải thêm detector riêng.
## Tối ưu hóa
- Resize ảnh nhỏ hơn để đưa vào model => sửa `self.transforms` trong phương thức `__init__` của class `DataFactory` trong `upt/utils.py`.
- Chuyển model và input thành dạng half để tăng tốc độ xử lý.
 
 
