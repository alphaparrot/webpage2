<?php
if(isset($_POST['submit'])){
// Fetching variables of the form which travels in URL
$neighborhood = $_POST['name'];
$filename = str_replace(" ","_",$neighborhood);
$filename = str_replace("-","_",$filename);
if($neighborhood !=''&& $neighborhood !='placeholder')
{
//  To redirect form on a particular page
header("Location:".$filename.".html");
}
else{
?><span><?php echo "Please select a neighborhood!";?></span> <?php
}
}
?>