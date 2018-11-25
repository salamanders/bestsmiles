import com.google.api.gax.core.FixedCredentialsProvider
import com.google.auth.oauth2.GoogleCredentials
import com.google.cloud.vision.v1.*
import com.google.gson.GsonBuilder
import com.google.protobuf.ByteString
import mu.KotlinLogging
import java.io.File

private val logger = KotlinLogging.logger {}

private val gson = GsonBuilder()
        .setPrettyPrinting()
        .create()!!

private const val fileName = "DSC02953_cropped.JPG"


fun tryGoogleVision() {
    logger.info { "Hi Player 1." }

    val credentials = GoogleCredentials
            .fromStream(ClassLoader
                    .getSystemClassLoader()
                    .getResourceAsStream("serviceAccountKey.json"))!!

    val settings = ImageAnnotatorSettings.newBuilder()
            .setCredentialsProvider(FixedCredentialsProvider.create(credentials))
            .build()!!

    val requests = mutableListOf<AnnotateImageRequest>()

    val imgBytes = ByteString.readFrom(File(fileName).inputStream())!!
    val img = Image.newBuilder().setContent(imgBytes).build()!! // TODO: WebP?  Auto-compress to 10mb?  crop?
    val feat = Feature.newBuilder()
            .setType(Feature.Type.FACE_DETECTION)
            .setMaxResults(600)
            .build()!!
    val request = AnnotateImageRequest.newBuilder().addFeatures(feat).setImage(img).build()!!
    requests.add(request)

    ImageAnnotatorClient.create(settings)!!.use { client ->
        logger.info { "Sending request..." }

        val (errors, successes) = client.batchAnnotateImages(requests)!!.responsesList!!.filterNotNull().partition { it.hasError() }
        successes.forEachIndexed { idx, air ->
            logger.info { "# Success $idx found ${air.faceAnnotationsCount}" }

            File("result_$idx.json").writer().use { osw ->
                //gson.toJson(air.faceAnnotationsList.filterNotNull().map { it.BAD }, osw)

                air.faceAnnotationsList.filterNotNull().forEach {
                    it.writeDelimitedTo(File("").outputStream())
                }
            }
        }
        errors.forEach { logger.warn { "# Error: ${it.error}" } }
    }

}