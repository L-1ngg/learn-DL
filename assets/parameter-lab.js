const weight = document.querySelector('#weight');
const canvas = document.querySelector('#plot');
const ctx = canvas.getContext('2d');
function render() {
  const w = Number(weight.value);
  document.querySelector('#weight-value').value = w.toFixed(1);
  const inputs = [1, 2, 3];
  document.querySelector('#samples').innerHTML = inputs.map(x => `<tr><td>${x}</td><td>${2 * x}</td><td>${(w * x).toFixed(1)}</td><td>${((w * x - 2 * x) ** 2).toFixed(2)}</td></tr>`).join('');
  document.querySelector('#loss').value = (inputs.reduce((sum, x) => sum + (w * x - 2 * x) ** 2, 0) / inputs.length).toFixed(2);
  const px = x => 55 + x * 200;
  const py = y => 315 - y * 23;
  ctx.clearRect(0, 0, 720, 360);
  ctx.fillStyle = '#fafafa'; ctx.fillRect(0, 0, 720, 360);
  ctx.font = '20px sans-serif';
  for (let y = 0; y <= 12; y += 3) {
    ctx.strokeStyle = '#d8ddda'; ctx.beginPath(); ctx.moveTo(px(0), py(y)); ctx.lineTo(px(3), py(y)); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.fillText(String(y), 15, py(y) + 7);
  }
  for (let x = 0; x <= 3; x++) ctx.fillText(String(x), px(x) - 5, 345);
  ctx.strokeStyle = '#08786d'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(3), py(w * 3)); ctx.stroke();
  ctx.lineWidth = 1;
  for (const x of inputs) {
    ctx.fillStyle = '#b34c22'; ctx.beginPath(); ctx.arc(px(x), py(2 * x), 7, 0, 2 * Math.PI); ctx.fill();
  }
}
weight.addEventListener('input', render);
document.querySelector('#prediction-quiz').addEventListener('submit', event => {
  event.preventDefault();
  const answer = Number(document.querySelector('#answer').value);
  document.querySelector('#quiz-feedback').textContent = answer === 8
    ? '正确：2 × 4 = 8。参数保持不变，用新输入计算预测；这一步不需要知道目标值。'
    : '再试一次：把 w = 2 和输入值 4 代入“预测值 = w × 输入值”。这里不是加法。';
});
render();
