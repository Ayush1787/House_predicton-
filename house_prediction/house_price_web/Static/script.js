function validateForm() {
    let area = document.getElementById("area").value;

    if (area <= 0) {
        alert("Area must be greater than 0");
        return false;
    }
    return true;
}
