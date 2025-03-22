#!/bin/bash
echo "===== KIỂM TRA MÔ HÌNH ĐÃ LƯU ====="
find ./saved_models -type f -name "*.h5" | sort

echo ""
echo "===== KIỂM TRA TRẠNG THÁI TRAINING ====="
if [ -f "training_status.json" ]; then
    cat training_status.json | python -m json.tool 
else
    echo "Không tìm thấy file training_status.json"
fi

echo ""
echo "===== LOGS TRAINING GẦN NHẤT ====="
grep -i "train\|model\|lstm\|transformer\|cnn" training_logs.txt | tail -n 20
