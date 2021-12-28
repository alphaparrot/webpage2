<?php
// var_dump($_POST);
// Fetching variables of the form which travels in URL
if(isset($_POST['submit'])){
echo $_POST['county'];
$location = explode("|",$_POST['county']);
var_dump($location);
$state = $location[1];
$county = $location[0];
$state = str_replace(" ","_",$state);
$state = str_replace("&","and",$state);
$state = str_replace(",","",$state);
$state = str_replace("/","_",$state);
$state = str_replace("-","_",$state);
$state = str_replace("+","_",$state);
$county = str_replace(" ","_",$county);
$county = str_replace("&","and",$county);
$county = str_replace(",","",$county);
$county = str_replace("/","_",$county);
$county = str_replace("-","_",$county);
$county = str_replace("+","_",$county);
$filename = $state."_".$county
echo $filename.".html";
if($state !=''&& $state !='placeholder')
{
//  To redirect form on a particular page
header("Location:".$filename.".html");
}
else{
?><span><?php echo "Please select a state or territory, and a county!";?></span> <?php
}
}
?>