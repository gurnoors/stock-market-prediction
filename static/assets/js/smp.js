$(document).ready(() => {
    $("#results").css("display","none")
    $("#loading").hide()

})


function predict(){
    let amount = $("#amount")[0].value
    let period_value = $("#period_value")[0].value
    let period_multiplier = $("#period_multiplier option:selected").text();
    
    $("#loading").show()
    console.log(amount)
    console.log(period_value)
    console.log(period_multiplier)

    let x = parseInt(period_value)
    if(period_multiplier == 'Days'){
        x = x ;
    }else if(period_multiplier == 'Weeks'){
        x = x * 7;
    }else if(period_multiplier == 'Months'){
        x = x * 30;
    }else if(period_multiplier == 'Years'){
        x = x * 365;
    }
    const url="/data?money=" + amount + "&period=" + x;
    console.log(url)
    // data = {
    //             stocks: [
    //                     {'name': 'NVDA', 'quantity': 23, 'quantity': 34.45},
    //                     {'name': 'GOOG', 'quantity': 22, 'total_price': 45.45},
    //                     {'name': 'MSFT', 'quantity': 21, 'total_price': 64.57},
    //                 ],
    //             total: 45
    //         }


    // data = {"prophet": {"total": 45, "stocks": [{"total_price": 34.45, "name": "NVDA", "quantity": 23}, {"total_price": 45.45, "name": "GOOG", "quantity": 22}, {"total_price": 64.57, "name": "MSFT", "quantity": 21}]}, "lstm": {"total": 45, "_comment": "total price", "stocks": [{"total_price": 34.45, "name": "NVDA", "quantity": 23}, {"total_price": 45.45, "name": "GOOG", "quantity": 22}, {"total_price": 64.57, "name": "MSFT", "quantity": 21}]}, "somethingelse": {}, "sarima": {}}

    // setTimeout(()=>{
    $.get(url, (data) => {
        console.log(data)
        y = data
        // data = data['prophet']
        $("#loading").hide()

        let table = $("#table tbody");
        $.each(data['prophet']['stocks'], function(idx, elem){
            table.append("<tr><td>"+elem.name+"</td><td>"+elem.quantity+"</td>   <td>"+elem.total_price+"$</td></tr>");
        });

        let total = $("#total")[0];
        total.innerHTML = data['prophet']['total'] + '$'


        $("#results").css("display","block")
    })
    // }, 10);
}

console.log("dffd")