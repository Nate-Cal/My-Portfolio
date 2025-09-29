hideBMI();
hideBuildMuscle();
hideLoseWeight();
hideBuildEndurance();
function BMIcalculatorImperial (form) {
    let feet = form.feetInput.value;
    let inches = form.inchesInput.value;
    let pounds =form.poundsInput.value;
    let displayMessage = "Please fill out the entire form to calculate your BMI.";

    let invalidInput = isEmpty(feet) || isEmpty(inches) || isEmpty(pounds);

    if (!invalidInput) {
        feet = Number(feet);
        inches = Number(inches);
        pounds = Number(pounds);
        let height = feet * 12 + inches;
        console.log(typeof(height))
        let BMI = (pounds) / (height ** 2) * 703; 
        BMI = BMI.toFixed(1);

        displayMessage = `Your BMI is ${BMI}!`;
    }

    document.getElementById("BMIvalue").innerHTML = displayMessage;
}
function BMIcalculatorMetric (form) {
    let centimeters = form.centimeterInput.value;
    let kilograms = form.kilogramInput.value;
    let displayMessage = "Please fill out the entire form to calculate your BMI";

    let invalidInput = isEmpty(centimeters) || isEmpty(kilograms);

    if (!invalidInput) {
        centimeters = Number(centimeters);
        kilograms = Number(kilograms);
        let meters = centimeters / 100;
        let BMI = (kilograms) / (meters ** 2);
        BMI = BMI.toFixed(1);

        displayMessage =  `Your BMI is ${BMI}!`;
    }
        
    document.getElementById("BMIvalue").innerHTML = displayMessage;
    }
function stepCounter (form) {
    let steps = form.stepInput.value;
    let displayMessage = "You have not walked any steps today";

    if (!isEmpty(steps)) {
        displayMessage = `You walked ${steps} steps!`;
    }
    document.getElementById("stepValue").innerHTML = displayMessage;
}
function showImperial(){
    BMIformImperial.style.display = "block";
    BMIformMetric.style.display = "none";
}
function showMetric(){
    BMIformImperial.style.display = "none";
    BMIformMetric.style.display = "block";
}
function hideBMI(){
    BMIformImperial.style.display = "none";
    BMIformMetric.style.display = "none";
}
function showBMI(){
    BMIformImperial.style.display = "block";
    BMIformMetric.style.display = "none";
}
function getRadioValue(){
    if(document.getElementById("buildMuscle").checked){
        displayBuildMuscle();
    }
    else if(document.getElementById("loseWeight").checked){
        displayLoseWeight();
    }
    else if(document.getElementById("buildEndurance").checked){
        displayBuildEndurance();
    }
}
function hideBuildMuscle(){
    var buildMuscleVideos = document.getElementById("buildMuscleVideos");
    buildMuscleVideos.style.display = "none";
}
function displayBuildMuscle(){
    var buildMuscleVideos = document.getElementById("buildMuscleVideos");
    buildMuscleVideos.style.display = "block";
    hideLoseWeight();
    hideBuildEndurance();
}
function hideLoseWeight(){
    var loseWeightVideos = document.getElementById("loseWeightVideos");
    loseWeightVideos.style.display = "none";
}
function displayLoseWeight(){
    var loseWeightVideos = document.getElementById("loseWeightVideos");
    loseWeightVideos.style.display = "block";
    hideBuildMuscle();
    hideBuildEndurance();    
}
function hideBuildEndurance(){
    var buildEnduraceVideos = document.getElementById("buildEnduranceVideos");
    buildEnduranceVideos.style.display = "none";
}
function displayBuildEndurance(){
    var buildEnduraceVideos = document.getElementById("buildEnduranceVideos");
    buildEnduranceVideos.style.display = "block";
    hideBuildMuscle();
    hideLoseWeight();
}
function isEmpty(input){
    return input === "" || input === undefined || input === NaN || input === "0" ? true : false;
}
