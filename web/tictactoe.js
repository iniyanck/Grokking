let session;
let boardState = Array(9).fill(0); // 0=Empty, 1=X, 2=O
let currentPlayer = 1; // 1=X starts

async function loadModel() {
    try {
        session = await ort.InferenceSession.create('../results/model_tictactoe.onnx');
        document.getElementById('status').innerText = "Model Loaded | Ready";
        runInference();
    } catch (e) {
        document.getElementById('status').innerText = "Error loading model: " + e;
        console.error(e);
    }
}

function initBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    for (let i = 0; i < 9; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.onclick = () => makeMove(i);
        cell.id = `cell-${i}`;
        board.appendChild(cell);
    }
}

function makeMove(idx) {
    if (boardState[idx] !== 0) return;
    
    boardState[idx] = currentPlayer;
    const cell = document.getElementById(`cell-${idx}`);
    cell.innerText = currentPlayer === 1 ? 'X' : 'O';
    cell.classList.add(currentPlayer === 1 ? 'x' : 'o');
    
    currentPlayer = currentPlayer === 1 ? 2 : 1;
    runInference();
}

function resetBoard() {
    boardState = Array(9).fill(0);
    currentPlayer = 1;
    initBoard();
    runInference();
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

async function runInference() {
    if (!session) return;
    
    // Create BigInt tensor inputs
    // The vocab size is 3 (2 for O). 
    // Input shape is [1, 9]
    // The model expects Int64 (which translates to BigInt in JS for onnxruntime)
    
    const inputData = BigInt64Array.from(boardState.map(x => BigInt(x)));
    const tensor = new ort.Tensor('int64', inputData, [1, 9]);
    
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    
    // Output shape should be [1, 3] (logits)
    const logits = results.output.data;
    const probs = softmax(Array.from(logits));
    
    // Update UI
    // Index 0: Draw, 1: X Win, 2: O Win
    updateBar('prob-draw', 'bar-draw', probs[0]);
    updateBar('prob-x', 'bar-x', probs[1]);
    updateBar('prob-o', 'bar-o', probs[2]);
}

function updateBar(textId, barId, prob) {
    const percent = (prob * 100).toFixed(1);
    document.getElementById(textId).innerText = percent + "%";
    document.getElementById(barId).style.width = percent + "%";
}

initBoard();
loadModel();
