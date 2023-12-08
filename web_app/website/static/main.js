$(document).ready(function(){
    $("#btn").click(function(){
        document.getElementById("formFileLg").disabled = true;
        document.getElementById("btn").className = "spinner-border spinner-grow text-primary mt-3";
        document.getElementById("btn").innerText = "";
        json_data = JSON.stringify(myData.datasets[0].data)
        $.ajax({
            contentType: 'application/json',
            type: "get",
            data : {data: json_data},
            success: function(response){
                $('.lead').text('WEEKLY PREDICTIONS')
                myData.datasets[0].data = response.prediction1
                myData.datasets[1].data = response.prediction2
                myData.datasets[2].data = response.prediction3
                myData.datasets[3].data = response.prediction4
                myData.datasets[4].data = response.prediction5
                myData.datasets[5].data = response.prediction6

                myData.datasets[0].label = 'XGBoost'
                myData.datasets[1].label = 'ExpSmooth'
                myData.datasets[2].label = 'Arima'
                myData.datasets[3].label = 'Univariate'
                myData.datasets[4].label = 'Synthetic'
                myData.datasets[5].label = 'Hybrid'

                myData.labels = ["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7"]
                
                myChart.update();
                document.getElementById("btn").className = "btn btn-outline-primary mt-3";
                document.getElementById("btn").innerText = "Predict!";
                document.getElementById("btn").style.visibility = "hidden"
                document.getElementById("formFileLg").disabled = false;
            }
        })
    })
})

$(document).ready(function(){

    function readSingleFile(e) {
        var file = e.target.files[0];
        if (!file) {
            myData.datasets.forEach((dataset) => {
                dataset.data = [];
                dataset.label ='';
            });
            myData.datasets[0].data = zeroData
            myData.datasets[0].label = ''
            myChart.update();
            $('.lead').text('UPLOAD DATA FILE (.csv)')
            document.getElementById("btn").style.visibility = "hidden"
            return;
        }
        var reader = new FileReader();
        reader.onload = function(e) {
            var contents = e.target.result.split("\r\n");
            contents.pop();
            numeric = true
            Object.keys(contents).forEach(key => {
                if (contents[key].match(/^[0-9]+$/) == null) { //if string has letter
                    numeric =false
                }
            });
            if (contents.length == 20 && numeric == true){
                displayContents(contents);
            } else { 
                myData.datasets.forEach((dataset) => {
                    dataset.data = [];
                    dataset.label ='';
                });
                myData.datasets[0].data = zeroData
                myData.datasets[0].label = ''
                myChart.update();
                $('.lead').text('FILE MUST HAVE 20 NUMERIC VALUES ON A COLUMN.')
                document.getElementById("btn").style.visibility = "hidden"
            }
        };
        reader.readAsText(file);
    }

    function displayContents(contents) {
        myData.datasets.forEach((dataset) => {
            dataset.data = [];
            dataset.label ='';
        });

        myData.datasets[0].data = contents

        myData.labels = ["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7","Day8", "Day9", "Day10",
        "Day11", "Day12", "Day13", "Day14","Day15", "Day16", "Day17", "Day18", "Day19", "Day20"]

        myData.datasets[0].label = 'Uploaded data'

        myChart.update();

        $('.lead').text('FILE LOADED')
        document.getElementById("btn").style.visibility = "visible"
    }

    const x = document.querySelector(".form-control")
    x.addEventListener("change", readSingleFile, false);

})