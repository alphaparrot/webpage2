<?php
// var_dump($_POST);
// Fetching variables of the form which travels in URL
if(isset($_POST['submit'])){
$phu = $_POST['phu'];
$filename = 'ontario_'.str_replace(" ","_",$phu);
$filename = str_replace('"','',$filename);
$filename = str_replace("&","and",$filename);
$filename = str_replace(",","",$filename);
$filename = str_replace("/","_",$filename);
$filename = str_replace("-","_",$filename);
$filename = str_replace("+","_",$filename);
// echo $filename.".html";
if($phu !=''&& $phu !='placeholder')
{
//  To redirect form on a particular page
header("Location:".$filename.".html");
}
else{
?><span><?php echo "Please select a public health unit!";?></span> <?php
}
}
?>