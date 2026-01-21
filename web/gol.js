let session;
let boardState = Array(25).fill(0); // 0=Dead, 1=Alive

async function loadModel() {
    try {
        session = await ort.InferenceSession.create('../results/model_gol.onnx');
        document.getElementById('status').innerText = "Model Loaded | Ready";
    } catch (e) {
        document.getElementById('status').innerText = "Error loading model: " + e;
        console.error(e);
    }
    runInference();
}

function initBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    for (let i = 0; i < 25; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell dead';
        cell.onclick = () => toggleCell(i);
        cell.id = `cell-${i}`;

        // Highlight center
        if (i === 12) {
            cell.style.borderColor = "#bb86fc";
            cell.style.borderWidth = "2px";
        }

        board.appendChild(cell);
    }
}

function toggleCell(idx) {
    boardState[idx] = boardState[idx] === 0 ? 1 : 0;
    updateVisuals();
    runInference();
}

function updateVisuals() {
    for (let i = 0; i < 25; i++) {
        const cell = document.getElementById(`cell-${i}`);
        if (boardState[i] === 1) {
            cell.classList.remove('dead');
            cell.classList.add('alive');
        } else {
            cell.classList.remove('alive');
            cell.classList.add('dead');
        }
    }
}

function clearBoard() {
    boardState = Array(25).fill(0);
    updateVisuals();
    runInference();
}

function randomize() {
    boardState = boardState.map(() => Math.random() > 0.5 ? 1 : 0);
    updateVisuals();
    runInference();
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// Actual GoL Logic for center cell (idx 12)
function calculateRule() {
    // Center is 12 (row 2, col 2)
    // Neighbors:
    // 6, 7, 8
    // 11,    13
    // 16, 17, 18
    const neighbors = [6, 7, 8, 11, 13, 16, 17, 18];
    const liveCount = neighbors.reduce((sum, idx) => sum + boardState[idx], 0);
    const center = boardState[12];

    if (center === 1) {
        if (liveCount < 2 || liveCount > 3) return 0;
        return 1;
    } else {
        if (liveCount === 3) return 1;
        return 0;
    }
}

async function runInference() {
    // 1. Calculate Actual Rule
    const actualNext = calculateRule();
    const ruleEl = document.getElementById('pred-rule');
    ruleEl.innerText = actualNext === 1 ? 'Alive' : 'Dead';
    ruleEl.className = actualNext === 1 ? 'cell alive' : 'cell dead';

    // 2. Run Model
    if (!session) return;

    const inputData = BigInt64Array.from(boardState.map(x => BigInt(x)));
    const tensor = new ort.Tensor('int64', inputData, [1, 25]);

    const feeds = { input: tensor };
    const results = await session.run(feeds);

    const logits = results.output.data;
    const probs = softmax(Array.from(logits));

    // Prob of being Alive (Class 1)
    const probAlive = probs[1];
    const isAlive = probAlive > 0.5;

    const modelEl = document.getElementById('pred-model');
    modelEl.innerText = isAlive ? `Alive (${(probAlive * 100).toFixed(0)}%)` : `Dead (${(probs[0] * 100).toFixed(0)}%)`;
    modelEl.className = isAlive ? 'cell alive' : 'cell dead';

    // Check if correct
    if (isAlive === (actualNext === 1)) {
        modelEl.style.border = "2px solid #03dac6"; // Match
    } else {
        modelEl.style.border = "2px solid #cf6679"; // Mismatch
    }
}

initBoard();
loadModel();
