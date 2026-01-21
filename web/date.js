let session;
const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

async function loadModel() {
    try {
        session = await ort.InferenceSession.create('../results/model_date.onnx');
        document.getElementById('status').innerText = "Model Loaded | Ready";
    } catch (e) {
        document.getElementById('status').innerText = "Error loading model (Make sure you trained 'date' dataset): " + e;
        console.error(e);
    }
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

async function predict() {
    const dateStr = document.getElementById('dateInput').value;
    if (!dateStr) return;

    const dateObj = new Date(dateStr);
    const y = dateObj.getFullYear();
    const m = dateObj.getMonth() + 1; // 1-12
    const d = dateObj.getDate();

    // JS getDay(): 0=Sun, 1=Mon...6=Sat
    // Our Dataset: 0=Mon, ... 5=Sat, 6=Sun
    // Mapping:
    const jsDay = dateObj.getDay(); // 0(Sun)..6(Sat)
    // Convert to our format (0=Mon...6=Sun)
    // If Sun(0) -> 6
    // Else -> jsDay - 1
    let actualLabel = jsDay === 0 ? 6 : jsDay - 1;

    // Display Actul
    document.getElementById('day-actual').innerText = days[actualLabel];

    // Inference
    if (!session) {
        alert("Model not loaded yet or failed to load.");
        return;
    }

    // Input: [1, 3] tensor (Year, Month, Day)
    // Wait, the dataset uses `(y, m, d)`
    // The model expects `Long` (Int64)

    const inputData = BigInt64Array.from([BigInt(y), BigInt(m), BigInt(d)]);
    const tensor = new ort.Tensor('int64', inputData, [1, 3]);

    try {
        const results = await session.run({ input: tensor });
        const logits = results.output.data;
        const probs = softmax(Array.from(logits));

        // Find max
        let maxIdx = 0;
        let maxP = -1;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxP) {
                maxP = probs[i];
                maxIdx = i;
            }
        }

        // Display Model
        document.getElementById('day-model').innerText = days[maxIdx];
        document.getElementById('conf-model').innerText = (maxP * 100).toFixed(1) + "%";

        // Visual Feedback
        const modelCard = document.getElementById('card-model');
        const actCard = document.getElementById('card-actual');

        modelCard.className = "day-card " + (maxIdx === actualLabel ? "correct" : "incorrect");
        actCard.className = "day-card neutral";

    } catch (e) {
        document.getElementById('status').innerText = "Inference Error: " + e;
    }
}

// Set today
document.getElementById('dateInput').valueAsDate = new Date();

loadModel();
