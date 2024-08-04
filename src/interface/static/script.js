// https://stackoverflow.com/a/5092038/18121288
function update_clock()
{
    var now = new Date(); // current date
    months = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "June", "July", "Aug",
        "Sept", "Oct", "Nov", "Dec"
    ];

    time = now.getHours() + ':' + now.getMinutes(),

    // a cleaner way than string concatenation
    date = [
        now.getDate(), 
        months[now.getMonth()],
        now.getFullYear()
    ].join("-");

    // set the content of the element with the ID time to the formatted string
    document.getElementById("timedate").innerHTML = [date, time].join(", ");
    console.log([date, time].join(", "));

    // call this function again in 1000ms
    setTimeout(update_clock, 1000);
}
