$(document).ready(() => {
    $("#results").css("display","none")
    $("#loading").hide()

})

function predict(){
    const url=""
    let amount = $("#amount")[0].value
    let period_value = $("#period_value")[0].value
    let period_multiplier = $("#period_multiplier option:selected").text();
    
    $("#loading").show()
    console.log(amount)
    console.log(period_value)
    console.log(period_multiplier)

    data = {
                stocks: [
                        {'name': 'NVDA', 'quantity': 23, 'quantity': 34.45},
                        {'name': 'GOOG', 'quantity': 22, 'total_price': 45.45},
                        {'name': 'MSFT', 'quantity': 21, 'total_price': 64.57},
                    ],
                total: 45
            }

    setTimeout(()=>{
    // $.post(url, data, (data, status) => {
        $("#loading").hide()

        let table = $("#table tbody");
        $.each(data['stocks'], function(idx, elem){
            table.append("<tr><td>"+elem.name+"</td><td>"+elem.quantity+"</td>   <td>"+elem.quantity+"$</td></tr>");
        });

        let total = $("#total")[0];
        total.innerHTML = data['total'] + '$'


        $("#results").css("display","block")
    // })
    }, 1500);
}
