let base64String = "";
function imageUploaded() {
  var file = document.querySelector("input[type=file]")["files"][0];
  var reader = new FileReader();
  reader.onload = function () {
    base64String = reader.result.replace("data:", "").replace(/^.+,/, "");
    imageBase64Stringsep = base64String;
  };
  reader.readAsDataURL(file);
}

// hiển thị ra chuõi base64
function displayString() {
  $.ajax({
    url: "http://192.168.1.86:9999/add",
    type: "POST",
    data:{"data":base64String},
    success: function (result, status, erro) {
     console.log(result);
     alert(result);
    },
    error: function(req, status, error) {
        alert( error);
    }
  });

//   console.log(base64String);
}
