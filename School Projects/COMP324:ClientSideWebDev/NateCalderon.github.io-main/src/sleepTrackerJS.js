
let monthlySleepLogs = {};

function createMonthSelector() {
  const monthSelector = document.getElementById('monthSelector');
  const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  months.forEach((month, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = month;
    monthSelector.appendChild(option);
  });
}

function getCurrentYearMonth() {
  const monthSelector = document.getElementById('monthSelector');
  const month = parseInt(monthSelector.value, 10);
  const year = new Date().getFullYear();
  return `${year}-${month}`;
}

function createDayInputs() {
  const container = document.getElementById('dailySleepInputs');
  container.innerHTML = ''; 
  const month = parseInt(document.getElementById('monthSelector').value, 10);
  const year = new Date().getFullYear();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const yearMonthKey = getCurrentYearMonth();

  for (let day = 1; day <= daysInMonth; day++) {
    const input = document.createElement('input');
    input.type = 'number';
    input.name = `day${day}`;
    input.placeholder = `Day ${day} hours`;
    input.className = 'sleep-hours-input';
    input.min = "0";
    input.max = "24";
    // Repopulate saved data if available
    if (monthlySleepLogs[yearMonthKey] && monthlySleepLogs[yearMonthKey][day]) {
      input.value = monthlySleepLogs[yearMonthKey][day];
    }

    container.appendChild(input);
  }
}

function saveCurrentMonthData() {
  const inputs = document.querySelectorAll('#dailySleepInputs input');
  const yearMonthKey = getCurrentYearMonth();
  monthlySleepLogs[yearMonthKey] = {};

  inputs.forEach((input, index) => {
    if (input.value) {
      monthlySleepLogs[yearMonthKey][index + 1] = parseFloat(input.value);
    }
  });

  saveSleepData();
}

function calculateSleep() {
  saveCurrentMonthData(); 
  const yearMonthKey = getCurrentYearMonth();
  const monthlyData = Object.values(monthlySleepLogs[yearMonthKey] || {});
  const monthlyTotal = monthlyData.reduce((acc, val) => acc + val, 0);
  const average = monthlyData.length > 0 ? monthlyTotal / monthlyData.length : 0;

  document.getElementById('averageSleep').textContent = average.toFixed(2);
  document.getElementById('totalSleepMonth').textContent = monthlyTotal.toFixed(2);

  const yearlyTotal = Object.values(monthlySleepLogs).flatMap(obj => Object.values(obj))
          .reduce((acc, val) => acc + val, 0);
  document.getElementById('totalSleepYear').textContent = yearlyTotal.toFixed(2);
}


function showVideos() {
  const selectedOption = document.querySelector('input[name="videoType"]:checked').value;
  const container = document.getElementById('videoContainer');
  container.innerHTML = ''; 

  let videos = [];
  if (selectedOption === 'tips') {
    videos = [
      "https://www.youtube.com/embed/t0kACis_dJE",
      "https://www.youtube.com/embed/ZKNQ6gsW45M",
      "https://www.youtube.com/embed/YvqeWcPwd2o",
      "https://www.youtube.com/embed/YJTCdxkcDPI"
    ];
  } else if (selectedOption === 'relaxing') {
    videos = [
      "https://www.youtube.com/embed/yIQd2Ya0Ziw",
      "https://www.youtube.com/embed/prfZFyp4XZk",
      "https://www.youtube.com/embed/VJU7sdpELNA",
      "https://www.youtube.com/embed/v-N5pJ6VFYc"
    ];
  }

  videos.forEach(videoUrl => {
    const iframe = document.createElement('iframe');
    iframe.src = videoUrl;
    iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
    iframe.allowFullscreen = true;
    iframe.frameBorder = "0";
    iframe.width = "100%";
    iframe.height = "auto";
    iframe.style.maxWidth = "560px"; // Set a max-width if necessary
    iframe.style.maxHeight = "315px"; // Set a max-height if necessary
    container.appendChild(iframe);
  });
}


function logSleepGoal() {
const sleepGoalInput = document.getElementById('sleepGoalInput').value;
const message = `You plan to average ${sleepGoalInput} hours of sleep this month!`;
document.getElementById('sleepGoalMessage').textContent = message;
}

function saveSleepData() {
  localStorage.setItem('monthlySleepLogs', JSON.stringify(monthlySleepLogs));
}

function loadSleepData() {
  const data = localStorage.getItem('monthlySleepLogs');
  if (data) {
    monthlySleepLogs = JSON.parse(data);
  }
}

window.onload = function() {
  loadSleepData();
  createMonthSelector();
  document.getElementById('monthSelector').addEventListener('change', function() {
    saveCurrentMonthData(); // Save current month data before switching
    createDayInputs();
  });
  const currentMonth = new Date().getMonth();
  document.getElementById('monthSelector').value = currentMonth;
  createDayInputs();
  calculateSleep(); // Initial calculation based on loaded data
};