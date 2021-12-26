<?php
// var_dump($_POST);
// Fetching variables of the form which travels in URL
if(isset($_POST['submit'])){
$province = $_POST['province'];
$filename = str_replace(" ","_",$province);
$filename = str_replace("&","and",$filename);
$filename = str_replace(",","",$filename);
$filename = str_replace("/","_",$filename);
$filename = str_replace("-","_",$filename);
$filename = str_replace("+","_",$filename);
// echo $filename.".html";
if($province !=''&& $province !='placeholder')
{
//  To redirect form on a particular page
header("Location:".$filename.".html");
}
else{
?><span><?php echo "Please select a province or territory!";?></span> <?php
}
}
?>