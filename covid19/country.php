<?php
// var_dump($_POST);
// Fetching variables of the form which travels in URL
if(isset($_POST['submit'])){
$state = $_POST['country'];
$filename = str_replace(" ","_",$state);
$filename = str_replace("&","and",$filename);
$filename = str_replace(",","",$filename);
$filename = str_replace("/","_",$filename);
$filename = str_replace("-","_",$filename);
$filename = str_replace("+","_",$filename);
// echo $filename.".html";
if($state !=''&& $state !='placeholder')
{
//  To redirect form on a particular page
header("Location:".$filename."_country.html");
}
else{
?><span><?php echo "Please select a country!";?></span> <?php
}
}
?>