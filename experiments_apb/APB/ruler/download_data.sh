
SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}/data/synthetic/json"

echo "Downloading Paul Graham Essays"
python download_paulgraham_essay.py

echo "Downloading QA Dataset"
bash download_qa_dataset.sh
